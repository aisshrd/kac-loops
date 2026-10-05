# kac-loops — Fluctuation corrections to the Kac model: exact finite-N theory and one-loop rules

Code and data accompanying

> T. A. Tazhimov, A. R. Sharden, V. S. Zheltukhin,
> *Fluctuation corrections to kinetic equations from the large-deviation Hamiltonian:
> exact finite-N theory and one-loop rules for the Kac model*, submitted to Physical Review E (2026).

All computations are in Python (NumPy, SciPy, SymPy, Matplotlib). Units: collision rate λ = 1.

## Contents

| File | What it does | Produces |
|---|---|---|
| `scripts/kac_geom.py` | grid, collision maps, collision operator Q, Onsager operator K for the Kac model (shared library; also used by the companion geometry paper) | — |
| `scripts/levelC.py` | (1) exact finite-N moment systems of degree 4 and 6 (SymPy) and their eigenvalues γ₄(N), γ₆(N); (2) low-statistics Monte Carlo (superseded by `mc2.py`); (3) Lyapunov equation for the covariance Σ(t) along the Krook–Wu trajectory | `data/levelC.json` (section `lyap` used in Fig. 1c) |
| `scripts/deg8.py` | exact degree-8 moment system and eigenvalues (γ₈(N), two-particle modes) | printed |
| `scripts/deg10.py` | exact degree-10 moment system and eigenvalues (γ₁₀(N)) | printed |
| `scripts/oneloop.py` | vertices T, U in the Hermite basis, identity T = U, one-loop formula for γₙ(N) for n = 4, 6, 8, 10 and comparison with the exact coefficients (Table I) | `data/oneloop.json` |
| `scripts/mc2.py` | Gillespie simulation of the Kac process, N = 8, 16, 32, 64, relaxation of m₄(t) versus the exact finite-N solution (Table II, Fig. 1a,b) | `data/mc2.json` |
| `scripts/figC.py` | assembles Fig. 1 from `data/mc2.json` and `data/levelC.json` | `figures/fig_levelC.png` |

## Reproduce

```bash
pip install -r requirements.txt
cd scripts
python levelC.py      # ~2 min: exact systems (deg 4, 6) + Lyapunov; writes ../data/levelC.json
python deg8.py        # ~5 s
python deg10.py       # ~10 s
python oneloop.py     # ~30 s: Table I
python mc2.py         # ~2 min: Table II; writes ../data/mc2.json
python figC.py        # Fig. 1
```
All scripts are run from inside `scripts/` and write their output to `../data/` and `../figures/` (relative paths).

## Key results reproduced
- γ₄(N) = (1/4)(1 + 2/N), γ₆(N) = (3/8)(1 + 4/N), γ₈(N) = 29/64 + 3/(32N), γ₁₀(N) = 65/128 + 31/(32N) (exact, `levelC.py`, `deg8.py`, `deg10.py`)
- one-loop coefficients 0.5, 1.5, 0.09375, 0.96875 (`oneloop.py`), identical to the exact ones
- m₄(∞) = 3 + 3(m₄(0) − 3)/(N + 2)
- Monte Carlo rates 0.3115 / 0.2887 / 0.2630 / 0.2581 for N = 8 / 16 / 32 / 64 (`mc2.py`, statistical error ≈ 0.017 per m₄ point)

## License
MIT (see `LICENSE`). If you use this code, please cite the paper above and the Zenodo DOI of this repository.
