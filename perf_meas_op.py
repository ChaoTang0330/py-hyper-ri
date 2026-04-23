from typing import Dict
import os
import torch
import torch.distributed as dist
import numpy as np
from astropy.io import fits

# from .utils import gen_imaging_weight
from src.measurement_operator.pysrc.utils.io_hs import load_data_to_tensor_hs
from src.measurement_operator.pysrc.measOperator.meas_op_hs import MeasOpHyperspectral

def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None else default

def main():
    # torch.backends.fp32_precision = "ieee"
    # torch.backends.cuda.matmul.fp32_precision = "ieee"
    # torch.backends.cudnn.fp32_precision = "ieee"
    # torch.backends.cudnn.conv.fp32_precision = "ieee" # "tf32"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True

    # Bind process to one GPU
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    data = load_data_to_tensor_hs(
        uv_file_path="/home/dp330/dp330/dc-tang2/Small-scale-RI-imaging-hs/data/3c353_uvw_id_00_dt_9.46_6.21_freqratio_1.70_nfreq_36_rotation_330.40",
        super_resolution=1.5,
        image_pixel_size=None,
        data_weighting=True,
        load_weight=False,
        img_size=(512, 512),
        weight_type="briggs",
        weight_gridsize=2,
        weight_robustness=0,
        dtype=torch.float64,
        device=device,
        verbose=True,
        ch_start=28,
        ch_last=36,
        ch_effctive=9,
        invert_uvw_flag=False,
    )

    # prepare measurement operator
    meas_op = MeasOpHyperspectral(
        u=data["u"],
        v=data["v"],
        img_size=(512, 512),
        natural_weight=data["nW"],
        image_weight=data["nWimag"],
        multi_stream=False,
        num_streams=3,
        device=device,
        dtype=torch.float64,
    )

    _ = meas_op.get_op_norm(rel_tol=1e-7, max_iter=200, compute_flag=True, verbose=True)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    _ = meas_op.get_op_norm(rel_tol=1e-7, max_iter=200, compute_flag=True, verbose=True)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_s = start_event.elapsed_time(end_event) / 1e3
    print(f"Time taken to compute operator norm: {elapsed_time_s} s")

if __name__ == "__main__":
    main()