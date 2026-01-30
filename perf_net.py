#!/usr/bin/env python3
import os
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.distributed as dist

try:
    import onnx
    import onnx2torch
except Exception:
    onnx = None
    onnx2torch = None

Tensor = torch.Tensor


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None else default


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank_info() -> Tuple[int, int, int, int]:
    """(rank, world_size, local_rank, node_rank)"""
    if not _is_dist():
        return 0, 1, 0, 0
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    node_rank = int(os.environ.get("NODE_RANK", os.environ.get("SLURM_NODEID", "0")))
    return rank, world, local_rank, node_rank


@dataclass
class Cfg:
    rank_num_ch: int = 9
    H: int = 512
    W: int = 512
    B: int = 1
    dtype: torch.dtype = torch.float32

    rand_trans: bool = True
    shared_trans: bool = True  # one flip/rot shared by all channels

    warmup: int = 10
    iters: int = 50


# ----------------------------
# transforms (CPU-sampled)
# ----------------------------
def _sample_transforms(n: int, shared: bool) -> Tuple[List[Tuple[int, ...]], List[int]]:
    flip_table = [tuple(), (2,), (3,), (2, 3)]
    if shared:
        fi = int(torch.randint(0, 4, (1,), device="cpu").item())
        rk = int(torch.randint(0, 4, (1,), device="cpu").item())
        flips = [flip_table[fi]] * n
        rots = [rk] * n
        return flips, rots

    flip_idx = torch.randint(0, 4, (n,), device="cpu").tolist()
    rot = torch.randint(0, 4, (n,), device="cpu").tolist()
    flips = [flip_table[i] for i in flip_idx]
    return flips, rot


def _apply_transform(x: Tensor, flip_dim: Tuple[int, ...], rot_k: int) -> Tensor:
    y = x
    if len(flip_dim) > 0:
        y = torch.flip(y, dims=flip_dim)
    if rot_k != 0:
        y = torch.rot90(y, k=rot_k, dims=(2, 3))
    return y.contiguous()


def _invert_transform(x: Tensor, flip_dim: Tuple[int, ...], rot_k: int) -> Tensor:
    y = x
    if rot_k != 0:
        y = torch.rot90(y, k=-rot_k, dims=(2, 3))
    if len(flip_dim) > 0:
        y = torch.flip(y, dims=flip_dim)
    return y.contiguous()


# ----------------------------
# model + per-channel params
# ----------------------------
def _load_net(onnx_path: str, device: torch.device, do_compile: bool, compile_mode: str) -> torch.nn.Module:
    if onnx2torch is None or onnx is None:
        raise RuntimeError("onnx and onnx2torch are required to load an ONNX model.")
    m = onnx.load(onnx_path)
    net = onnx2torch.convert(m)
    net.eval().to(device)
    if do_compile:
        # best for inference on fixed shapes
        net = torch.compile(net, mode=compile_mode, dynamic=False)
    return net


def _prepare_per_channel_tensors(device: torch.device, dtype: torch.dtype, C: int) -> Tuple[List[Tensor], List[Tensor]]:
    scalings: List[Tensor] = []
    spec_w: List[Tensor] = []
    for i in range(C):
        s = torch.tensor(1.0 + 0.02 * i, device=device, dtype=dtype).view(1, 1, 1, 1)
        scalings.append(s)
        # keep this realistic: nonzero but small
        w = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 3, 1, 1)
        spec_w.append(w)
    return scalings, spec_w


# ----------------------------
# build 3-ch net inputs (wrapper cost included)
# ----------------------------
def _build_net_inputs(x: Tensor, xL: Tensor, xR: Tensor) -> List[Tensor]:
    C = x.shape[1]
    out: List[Tensor] = [None] * C  # type: ignore
    for i in range(C):
        if i == 0:
            out[i] = torch.cat([xL, x[:, 0:2, :, :]], dim=1).contiguous()
        elif i == C - 1:
            out[i] = torch.cat([x[:, C - 2:C, :, :], xR], dim=1).contiguous()
        else:
            out[i] = x[:, i - 1 : i + 2, :, :].contiguous()
    return out


# ----------------------------
# comm primitives
# ----------------------------
@torch.no_grad()
def _exchange_borders_sync_wait(x: Tensor, device: torch.device) -> Tuple[Tensor, Tensor]:
    """Blocking correctness version: batch_isend_irecv + req.wait()."""
    if not _is_dist() or dist.get_world_size() == 1:
        return x[:, :1].clone(), x[:, -1:].clone()

    rank = dist.get_rank()
    world = dist.get_world_size()

    x_to_left = x[:, :1, :, :].contiguous().clone()
    x_to_right = x[:, -1:, :, :].contiguous().clone()
    x_from_left = torch.empty_like(x_to_left, device=device)
    x_from_right = torch.empty_like(x_to_right, device=device)

    ops: List[dist.P2POp] = []
    if rank == 0:
        x_from_left.copy_(x_to_left)
    else:
        ops.append(dist.P2POp(dist.isend, x_to_left, rank - 1))
        ops.append(dist.P2POp(dist.irecv, x_from_left, rank - 1))

    if rank == world - 1:
        x_from_right.copy_(x_to_right)
    else:
        ops.append(dist.P2POp(dist.isend, x_to_right, rank + 1))
        ops.append(dist.P2POp(dist.irecv, x_from_right, rank + 1))

    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for r in reqs:
            r.wait()

    return x_from_left, x_from_right


@torch.no_grad()
def _exchange_borders_async_on_stream(
    x: Tensor,
    device: torch.device,
    stream_comm_left: torch.cuda.Stream,
    stream_comm_right: torch.cuda.Stream,
    curr_stream: torch.cuda.Stream,
) -> Tuple[Tensor, Tensor, Optional[torch.cuda.Event], Optional[torch.cuda.Event]]:
    """
    Async version: enqueue comm on comm streams (left/right). No req.wait().
    Returns xL/xR and end-events for comm streams (to measure comm time or to join).
    """
    if not _is_dist() or dist.get_world_size() == 1:
        return x[:, :1].clone(), x[:, -1:].clone(), None, None

    rank = dist.get_rank()
    world = dist.get_world_size()

    x_to_left = x[:, :1, :, :].contiguous().clone()
    x_to_right = x[:, -1:, :, :].contiguous().clone()
    x_from_left = torch.empty_like(x_to_left, device=device)
    x_from_right = torch.empty_like(x_to_right, device=device)

    evL_end: Optional[torch.cuda.Event] = None
    evR_end: Optional[torch.cuda.Event] = None

    # left
    if rank == 0:
        x_from_left.copy_(x_to_left)
    else:
        stream_comm_left.wait_stream(curr_stream)
        with torch.cuda.stream(stream_comm_left):
            ops = [
                dist.P2POp(dist.isend, x_to_left, rank - 1),
                dist.P2POp(dist.irecv, x_from_left, rank - 1),
            ]
            _ = dist.batch_isend_irecv(ops)
            evL_end = torch.cuda.Event(enable_timing=True)
            evL_end.record(stream_comm_left)

    # right
    if rank == world - 1:
        x_from_right.copy_(x_to_right)
    else:
        stream_comm_right.wait_stream(curr_stream)
        with torch.cuda.stream(stream_comm_right):
            ops = [
                dist.P2POp(dist.isend, x_to_right, rank + 1),
                dist.P2POp(dist.irecv, x_from_right, rank + 1),
            ]
            _ = dist.batch_isend_irecv(ops)
            evR_end = torch.cuda.Event(enable_timing=True)
            evR_end.record(stream_comm_right)

    return x_from_left, x_from_right, evL_end, evR_end


# ----------------------------
# denoise core (mimic)
# ----------------------------
@torch.no_grad()
def _denoise_core(
    net_input: Tensor,
    net: torch.nn.Module,
    net_scaling: Tensor,
    spec_corr_w: Tensor,
    rand_trans: bool,
    flip_dim: Tuple[int, ...],
    rot_k: int,
) -> Tensor:
    x = net_input
    if rand_trans:
        x = _apply_transform(x, flip_dim, rot_k)

    out = net(x / net_scaling) * net_scaling

    if rand_trans:
        out = _invert_transform(out, flip_dim, rot_k)

    out = out + torch.sum(net_input * spec_corr_w, dim=1, keepdim=True)
    return out


# ----------------------------
# Bench variants
# ----------------------------
@torch.no_grad()
def bench_wrapper_sequential(
    net: torch.nn.Module,
    x: Tensor,
    xL: Tensor,
    xR: Tensor,
    scalings: List[Tensor],
    spec_w: List[Tensor],
    cfg: Cfg,
) -> Tensor:
    C = cfg.rank_num_ch
    flips, rots = _sample_transforms(C, cfg.shared_trans) if cfg.rand_trans else ([tuple()] * C, [0] * C)
    net_inputs = _build_net_inputs(x, xL, xR)

    y = torch.empty_like(x)
    for i in range(C):
        out = _denoise_core(net_inputs[i], net, scalings[i], spec_w[i], cfg.rand_trans, flips[i], rots[i])
        y[:, i : i + 1].copy_(torch.clamp(out, min=0.0))
    return y


@torch.no_grad()
def bench_wrapper_batched_bestcase(
    net: torch.nn.Module,
    x: Tensor,
    xL: Tensor,
    xR: Tensor,
    scalings: List[Tensor],
    spec_w: List[Tensor],
    cfg: Cfg,
) -> Tensor:
    """
    Best-case batching: one net shared by all channels.
    - shared transform across channels (optional)
    - per-channel scaling & spec weights still applied
    """
    C = cfg.rank_num_ch
    flips, rots = _sample_transforms(C, cfg.shared_trans) if cfg.rand_trans else ([tuple()] * C, [0] * C)
    net_inputs = _build_net_inputs(x, xL, xR)

    # apply transform per channel (unavoidable if transforms differ, but here shared_trans may reduce randomness variance)
    x_trans_list: List[Tensor] = []
    for i in range(C):
        inp = net_inputs[i]
        if cfg.rand_trans:
            inp = _apply_transform(inp, flips[i], rots[i])
        else:
            inp = inp.contiguous()
        x_trans_list.append(inp)

    x_trans = torch.cat(x_trans_list, dim=0)  # (C*B,3,H,W)

    # per-sample scaling: (C*B,1,1,1)
    s_batch = torch.cat([scalings[i].expand(cfg.B, 1, 1, 1) for i in range(C)], dim=0)

    out_trans = net(x_trans / s_batch) * s_batch  # (C*B,1,H,W)

    # invert + spec corr + clamp
    y = torch.empty_like(x)
    for i in range(C):
        out_i = out_trans[i * cfg.B : (i + 1) * cfg.B]  # (B,1,H,W)
        if cfg.rand_trans:
            out_i = _invert_transform(out_i, flips[i], rots[i])
        out_i = out_i + torch.sum(net_inputs[i] * spec_w[i], dim=1, keepdim=True)
        y[:, i : i + 1].copy_(torch.clamp(out_i, min=0.0))
    return y


@torch.no_grad()
def bench_wrapper_multistream(
    net: torch.nn.Module,
    x: Tensor,
    xL: Tensor,
    xR: Tensor,
    scalings: List[Tensor],
    spec_w: List[Tensor],
    cfg: Cfg,
    streams_ch: List[torch.cuda.Stream],
    curr_stream: torch.cuda.Stream,
) -> Tensor:
    C = cfg.rank_num_ch
    flips, rots = _sample_transforms(C, cfg.shared_trans) if cfg.rand_trans else ([tuple()] * C, [0] * C)
    net_inputs = _build_net_inputs(x, xL, xR)

    y = torch.empty_like(x)
    for i in range(C):
        streams_ch[i].wait_stream(curr_stream)
        with torch.cuda.stream(streams_ch[i]):
            out = _denoise_core(net_inputs[i], net, scalings[i], spec_w[i], cfg.rand_trans, flips[i], rots[i])
            y[:, i : i + 1].copy_(torch.clamp(out, min=0.0))

    for s in streams_ch:
        curr_stream.wait_stream(s)

    return y


# ----------------------------
# timing helpers
# ----------------------------
def _time_cuda(fn, warmup: int, iters: int, device: torch.device, *args) -> float:
    for _ in range(warmup):
        _ = fn(*args)
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = fn(*args)
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end)
    return (ms / 1e3) / iters


def _time_comm_only_sync(cfg: Cfg, x: Tensor, device: torch.device) -> float:
    def _comm():
        _ = _exchange_borders_sync_wait(x, device)
        return None
    return _time_cuda(_comm, cfg.warmup, cfg.iters, device)


def _time_comm_only_async(cfg: Cfg, x: Tensor, device: torch.device,
                          streamL: torch.cuda.Stream, streamR: torch.cuda.Stream) -> float:
    curr = torch.cuda.current_stream(device=device)

    def _comm():
        _, _, evL, evR = _exchange_borders_async_on_stream(x, device, streamL, streamR, curr)
        # ensure comm finishes before returning (comm-only timing)
        if evL is not None:
            evL.synchronize()
        if evR is not None:
            evR.synchronize()
        return None

    return _time_cuda(_comm, cfg.warmup, cfg.iters, device)


# ----------------------------
# full workflow timing (distributed)
# ----------------------------
@torch.no_grad()
def bench_full_sync_wait(
    net: torch.nn.Module,
    x: Tensor,
    scalings: List[Tensor],
    spec_w: List[Tensor],
    cfg: Cfg,
    device: torch.device,
) -> Tensor:
    xL, xR = _exchange_borders_sync_wait(x, device)
    return bench_wrapper_sequential(net, x, xL, xR, scalings, spec_w, cfg)


@torch.no_grad()
def bench_full_overlap_async_edge_wait(
    net: torch.nn.Module,
    x: Tensor,
    scalings: List[Tensor],
    spec_w: List[Tensor],
    cfg: Cfg,
    device: torch.device,
    stream_comm_left: torch.cuda.Stream,
    stream_comm_right: torch.cuda.Stream,
    streams_ch: List[torch.cuda.Stream],
) -> Tensor:
    """
    Async comm on comm streams; compute streams:
      - interior channels run immediately
      - edge channel streams wait on comm stream (for correctness)
      - join all streams back to current at end
    """
    C = cfg.rank_num_ch
    curr = torch.cuda.current_stream(device=device)

    xL, xR, _, _ = _exchange_borders_async_on_stream(
        x, device, stream_comm_left, stream_comm_right, curr
    )

    flips, rots = _sample_transforms(C, cfg.shared_trans) if cfg.rand_trans else ([tuple()] * C, [0] * C)

    # build net_inputs on current stream (wrapper cost included)
    net_inputs = _build_net_inputs(x, xL, xR)

    y = torch.empty_like(x)

    for i in range(C):
        streams_ch[i].wait_stream(curr)
        # edge correctness: wait on comm stream for recv completion
        if _is_dist() and dist.get_world_size() > 1:
            rank = dist.get_rank()
            world = dist.get_world_size()
            if i == 0 and rank > 0:
                streams_ch[i].wait_stream(stream_comm_left)
            if i == C - 1 and rank < world - 1:
                streams_ch[i].wait_stream(stream_comm_right)

        with torch.cuda.stream(streams_ch[i]):
            out = _denoise_core(net_inputs[i], net, scalings[i], spec_w[i], cfg.rand_trans, flips[i], rots[i])
            y[:, i : i + 1].copy_(torch.clamp(out, min=0.0))

    # join compute streams
    for s in streams_ch:
        curr.wait_stream(s)
    # join comm streams so next iteration can't clobber buffers (safe)
    curr.wait_stream(stream_comm_left)
    curr.wait_stream(stream_comm_right)

    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, default="/home/dp330/dp330/dc-tang2/Small-scale-RI-imaging-hs/airi_denoisers/shelf_cube_ds2_mix_noaug_curv_fix/dncnn_noskip_nch_1_l1_mix_origin_sigma_0.00016_ljr_8e-06_lr_1e-05_seed_0.onnx")
    ap.add_argument("--rank_num_ch", type=int, default=18)
    ap.add_argument("--H", type=int, default=512)
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)

    ap.add_argument("--rand_trans", action="store_true", help="enable flips/rots")
    ap.add_argument("--shared_trans", action="store_true", help="use one flip/rot for all channels")
    ap.add_argument("--no_shared_trans", action="store_true", help="force different trans per channel")

    ap.add_argument("--compile", action="store_true", help="torch.compile(net)")
    ap.add_argument("--compile_mode", type=str, default="max-autotune",
                    choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    args = ap.parse_args()

    # dist init (if env provided)
    world = env_int("WORLD_SIZE", 1)
    rank = env_int("RANK", 0)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world,
            device_id=torch.device("cuda:0"),
        )
    r, w, local_rank, node_rank = _rank_info()

    # bind to local GPU 0 (CUDA_VISIBLE_DEVICES should already be set by slurm wrapper)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # precision controls (keep deterministic-ish)
    torch.backends.cudnn.benchmark = True

    shared_trans = args.shared_trans and (not args.no_shared_trans)

    cfg = Cfg(
        rank_num_ch=args.rank_num_ch,
        H=args.H,
        W=args.W,
        B=args.batch,
        dtype=torch.float32,
        rand_trans=args.rand_trans,
        shared_trans=shared_trans,
        warmup=args.warmup,
        iters=args.iters,
    )

    if _is_dist():
        dist.barrier()

    if r == 0:
        print(
            f"=== denoise bench | world={w} rank={r} local_rank={local_rank} node_rank={node_rank} "
            f"| C={cfg.rank_num_ch} B={cfg.B} HxW={cfg.H}x{cfg.W} "
            f"| rand_trans={cfg.rand_trans} shared_trans={cfg.shared_trans} "
            f"| compile={args.compile}({args.compile_mode}) iters={cfg.iters} warmup={cfg.warmup} ===",
            flush=True,
        )

    # build one shared net (best-case)
    net = _load_net(args.onnx, device, args.compile, args.compile_mode)

    scalings, spec_w = _prepare_per_channel_tensors(device, cfg.dtype, cfg.rank_num_ch)

    # fixed input outside timing
    x = torch.rand(cfg.B, cfg.rank_num_ch, cfg.H, cfg.W, device=device, dtype=cfg.dtype)

    # persistent streams (fix)
    streams_ch = [torch.cuda.Stream(device=device) for _ in range(cfg.rank_num_ch)]
    stream_comm_left = torch.cuda.Stream(device=device)
    stream_comm_right = torch.cuda.Stream(device=device)

    # (optional) for wrapper-only benchmarks: use local borders (no comm)
    xL_local = x[:, :1].clone()
    xR_local = x[:, -1:].clone()

    if _is_dist():
        dist.barrier()
    torch.cuda.synchronize(device)

    # ------------------------------------------------------------
    # 1) COMM cost alone
    # ------------------------------------------------------------
    t_comm_sync = _time_comm_only_sync(cfg, x, device)
    if _is_dist():
        dist.barrier()
    t_comm_async = _time_comm_only_async(cfg, x, device, stream_comm_left, stream_comm_right)

    # ------------------------------------------------------------
    # 2) WRAPPER cost (single GPU, no comm)
    # ------------------------------------------------------------
    if _is_dist():
        dist.barrier()
    t_wrap_seq = _time_cuda(
        bench_wrapper_sequential, cfg.warmup, cfg.iters, device,
        net, x, xL_local, xR_local, scalings, spec_w, cfg
    )
    if _is_dist():
        dist.barrier()
    t_wrap_bat = _time_cuda(
        bench_wrapper_batched_bestcase, cfg.warmup, cfg.iters, device,
        net, x, xL_local, xR_local, scalings, spec_w, cfg
    )
    if _is_dist():
        dist.barrier()
    curr = torch.cuda.current_stream(device=device)
    t_wrap_ms = _time_cuda(
        bench_wrapper_multistream, cfg.warmup, cfg.iters, device,
        net, x, xL_local, xR_local, scalings, spec_w, cfg, streams_ch, curr
    )

    # ------------------------------------------------------------
    # 3) FULL workflow (with comm)
    # ------------------------------------------------------------
    if _is_dist():
        dist.barrier()
    t_full_sync = _time_cuda(
        bench_full_sync_wait, cfg.warmup, cfg.iters, device,
        net, x, scalings, spec_w, cfg, device
    )
    if _is_dist():
        dist.barrier()
    t_full_ovlp = _time_cuda(
        bench_full_overlap_async_edge_wait, cfg.warmup, cfg.iters, device,
        net, x, scalings, spec_w, cfg, device, stream_comm_left, stream_comm_right, streams_ch
    )

    # per-rank print
    print(
        f"[rank {r}] sec/call: "
        f"comm_sync={t_comm_sync:.6f} comm_async={t_comm_async:.6f} | "
        f"wrap_seq={t_wrap_seq:.6f} wrap_bat={t_wrap_bat:.6f} wrap_ms={t_wrap_ms:.6f} | "
        f"full_sync={t_full_sync:.6f} full_ovlp={t_full_ovlp:.6f}",
        flush=True,
    )

    # max across ranks (step time)
    if _is_dist():
        t = torch.tensor(
            [t_comm_sync, t_comm_async, t_wrap_seq, t_wrap_bat, t_wrap_ms, t_full_sync, t_full_ovlp],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        if r == 0:
            print(
                f"[MAX over ranks] sec/call: "
                f"comm_sync={t[0].item():.6f} comm_async={t[1].item():.6f} | "
                f"wrap_seq={t[2].item():.6f} wrap_bat={t[3].item():.6f} wrap_ms={t[4].item():.6f} | "
                f"full_sync={t[5].item():.6f} full_ovlp={t[6].item():.6f}",
                flush=True,
            )

    if _is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()