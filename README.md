# Cross-paradigm audit of LLM judges for clinical text evaluation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23581371.svg)](https://doi.org/10.5281/zenodo.23581371)

Code for auditing six open-weight LLM judges against expert ratings on four German clinical datasets across pointwise, pairwise and pass-or-fail evaluation.

## Usage

```bash
uv sync
cp .env.example .env   # judge endpoints (see serve.sh) and data directories
bash run_all.sh        # preprocessing, judge inference, metrics, figures and tables
```

The clinical datasets contain protected health information and are available from the corresponding author on reasonable request.

## Citation

Pakull, T. M. G. et al. Cross-paradigm audit of large language model judges for clinical text evaluation. *npj Digit. Med.* (2026). https://doi.org/10.1038/s41746-026-03443-2

Received: 2 June 2026 / Accepted: 13 October 2026

```bibtex
@article{pakull2026audit,
  title   = {Cross-paradigm audit of large language model judges for clinical text evaluation},
  author  = {Pakull, Tabea M. G. and Warmer, Sina and Arzideh, Kamyar and Hilser, Thomas and Hosters, Bernadette and Damm, Hendrik and Nensa, Felix and Schuler, Martin and Borys, Katarzyna and Horn, Peter A. and Hosch, Ren{\'e} and Friedrich, Christoph M.},
  journal = {npj Digital Medicine},
  year    = {2026},
  doi     = {10.1038/s41746-026-03443-2}
}
```
