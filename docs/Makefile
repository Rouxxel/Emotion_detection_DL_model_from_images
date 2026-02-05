# Minimal Makefile for Sphinx docs (source and build are in docs/ itself)
SPHINXBUILD = sphinx-build
SOURCEDIR = .
BUILDDIR = _build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
