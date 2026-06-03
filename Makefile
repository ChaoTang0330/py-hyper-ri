.PHONY: docs docs-clean docs-strict

PYTHON ?= python
SPHINXOPTS ?= --keep-going
SPHINXBUILD ?= $(PYTHON) -m sphinx
DOCS_SOURCE = docs
DOCS_BUILD = docs/_build/html

docs:
	$(SPHINXBUILD) $(SPHINXOPTS) -b html $(DOCS_SOURCE) $(DOCS_BUILD)

docs-clean:
	rm -rf $(DOCS_BUILD)
	$(SPHINXBUILD) $(SPHINXOPTS) -b html $(DOCS_SOURCE) $(DOCS_BUILD)

docs-strict:
	$(SPHINXBUILD) $(SPHINXOPTS) -W -b html $(DOCS_SOURCE) $(DOCS_BUILD)
