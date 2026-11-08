# What a dissipative quantum system remembers

Manuscript source, code and data for the paper by Markus Baumann, Itamar Fink, Johannes Wittmann,
Claudia Linnhoff-Popien and Jonas Stein, [arXiv:2608.14181](https://arxiv.org/abs/2608.14181) (version 2).

- [Read the paper](paper_v2/dissipation_qrc.pdf); its LaTeX source is in `paper_v2/`
- Data behind every figure: `paper_v2/data/snapshot.json`; raw records: `paper_v2/data/raw_results.tar.gz`
- Interval certificates of the three-spin limits (Appendix D): `paper_v2/certificates/spin_certificates.zip`,
  also attached to the PDF; its `README.txt` explains how to run them

Sections 3.1 to 3.3 of the paper use exact expectation values; Sec. 3.4 counts measurements, replaying the
inputs that the memory holds from a fixed state. Five-spin runs use eight fixed seeds unless the paper states
fewer.

## Reproduce

```bash
python -m pip install -r requirements.txt     # numpy, scipy, matplotlib, pytest; figures also need LaTeX
python paper_v2/figures/make_figures.py       # Figs. 2-7 from paper_v2/data/snapshot.json
cd paper_v2/figures && pdflatex fig1_concept.tex && cd ../..   # Fig. 1 (TikZ)
cd paper_v2 && latexmk -pdf dissipation_qrc.tex && cd ..       # the paper
```

To rebuild the snapshot from the raw records, unpack them at the repository root and export; this reproduces
`snapshot.json` byte for byte:

```bash
tar -xzf paper_v2/data/raw_results.tar.gz     # creates results/
python paper_v2/figures/export_snapshot.py
```

## Code

The raw records come from these scripts; each one documents its stages, and they run for minutes to hours.

| Script | Content |
|---|---|
| `experiments/run_dark_projection.py` | five-spin memory, spectra, decay rates and every decay rate of Fig. 2d, dark-state model, erasure, write along and across the phases, added local loss |
| `experiments/run_jump_families.py` | exchange, gain/loss and dephasing; protected operators |
| `experiments/run_review_runs.py` | gentle writing, slow modes up to large loss, local dephasing, signal size |
| `experiments/run_clock.py` | time to forget at four input amplitudes, finite sampling, budgets and crossings with the replayed inputs, write onto the slow modes, input on one spin |
| `experiments/run_five_spin_orders.py` | exact order distribution of the five-spin capacity against the mixing law |
| `experiments/run_leak_size.py` | leak coefficient of a shared channel at three to six spins |
| `experiments/run_spin_orders.py` | three-spin order shares at a fixed amplitude and with gentle writing; prefactor scan |
| `experiments/theory/order_law/reconstruct.py` | exact check of the mixing law (Fig. 5) |
| `experiments/referee_checks/` | training-set, prefactor and principal-direction checks |

The scripts share an independent Pauli-basis simulator and shot sampler
(`experiments/theory/finite_shots_vs_gamma/psim.py` and `pscore.py`), seeds and jump operators from
`experiments/run_final_scaling.py`, capacities from `experiments/run_ipc_wide.py`, and the reservoir library in
`src/qrc/`; `python -m pytest` runs the tests of that library.

## Cite

```bibtex
@misc{Baumann2026,
  title         = {What a dissipative quantum system remembers},
  author        = {Baumann, Markus and Fink, Itamar and Wittmann, Johannes and Linnhoff-Popien, Claudia and Stein, Jonas},
  year          = {2026},
  eprint        = {2608.14181},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2608.14181}
}
```

## License

MIT; see [LICENSE](LICENSE).
