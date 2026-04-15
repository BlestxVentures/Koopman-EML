# Koopman-EML

**Koopman Operator Lifting via EML Observable Dictionaries**

A GPU-optimized framework that uses the EML (Exp-Minus-Log) Sheffer operator as a complete, gradient-trainable, symbolically-interpretable dictionary for Koopman operator approximation.

## What is EML?

The EML operator ([Odrzywołek, 2026](https://arxiv.org/abs/2603.21852)) is the continuous-math analog of the NAND gate:

```
eml(x, y) = exp(x) - ln(y)
```

Paired with the constant `1`, it generates **every** standard elementary function.
The grammar is trivially simple: `S -> 1 | eml(S, S)`.

## Key Idea

We use parameterized EML trees as **learnable Koopman observables**:
- Each observable is an EML tree with Gumbel-softmax routing
- After training with temperature annealing, weights snap to one-hot
- The result is an **exact symbolic formula** for each observable
- This yields closed-form Koopman eigenfunctions -- fully interpretable

## Features

- **Taylor-series GPU primitives**: exp/ln via Horner-form polynomials (zero transcendental calls)
- **Vectorized tree evaluation**: all trees at all depths computed in batched operations
- **Three-phase training**: exploration -> hardening -> snap + fine-tune
- **PySINDy integration**: `EMLLibrary` as a drop-in feature library
- **CTF benchmarking**: evaluation against the [dynamicsai.org Common Task Framework](https://dynamicsai.org/CTF/)

## Installation

```bash
pip install -e ".[all]"
```

Or with specific extras:
```bash
pip install -e ".[sindy]"   # PySINDy integration
pip install -e ".[dev]"     # development tools
```

## Quick Start

```python
from koopman_eml import KoopmanEML
from koopman_eml.training import train_koopman_eml
from koopman_eml.analysis import koopman_eigendecomposition, extract_eml_formulas

model = KoopmanEML(state_dim=3, n_observables=16, tree_depth=3)
history = train_koopman_eml(model, X_k, X_k1, device="cuda")

eig = koopman_eigendecomposition(model)
formulas = extract_eml_formulas(model)
```

### PySINDy Integration

```python
import pysindy as ps
from koopman_eml.sindy import EMLLibrary

model = ps.SINDy(feature_library=EMLLibrary(depth=2, n_vars=3))
model.fit(X, t=dt)
model.print()
```

## Project Structure

```
src/koopman_eml/
    eml_ops.py          # Taylor primitives, EML operator
    eml_tree.py         # Parameterized EML trees
    koopman_model.py    # Full Koopman model
    training.py         # Three-phase training loop
    analysis.py         # Eigendecomposition, formula extraction
    sindy.py            # PySINDy EMLLibrary
    ctf.py              # CTF benchmark adapter
baselines/              # EDMD, Deep Koopman, PySR baselines
experiments/            # CTF Lorenz & KS experiments
```

## References

- Odrzywołek, A. (2026). *All elementary functions from a single operator.* [arXiv:2603.21852](https://arxiv.org/abs/2603.21852)
- Brunton et al. (2022). *Modern Koopman theory for dynamical systems.* SIAM Review.
- Brunton et al. (2016). *Discovering governing equations from data by sparse identification of nonlinear dynamical systems.* PNAS.
- Wyder et al. (2025). *Common Task Framework for Scientific ML.* NeurIPS Datasets & Benchmarks.

See [REFERENCES.md](REFERENCES.md) for the full bibliography.

## License

MIT -- see [LICENSE](LICENSE).
