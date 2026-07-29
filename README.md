<div align="center">

# Hard-constrained thermodynamic constitutive learning

Companion research code for learning inelastic constitutive models from
stress–strain histories while enforcing energy consistency, stability, and
non-negative dissipation by construction.

[Paper](https://arxiv.org/abs/2605.16837) ·
[Setup](#environment-setup) ·
[Examples](#included-experiments) ·
[Tests](#tests) ·
[Reproducibility notes](docs/REPRODUCIBILITY.md)

</div>

This repository accompanies:

> Filippo Masi, “Learning inelastic constitutive models from stress–strain
> data under hard thermodynamic constraints,” arXiv:2605.16837, 2026.

The implementation couples two neural parameterizations:

- a convex or partially input-convex free energy
  $`\psi_{\boldsymbol{\theta}}(\boldsymbol{s})`$, whose derivatives define
  stress and internal thermodynamic forces; and
- a state-dependent transport operator
  $`\mathbb{L}_{\boldsymbol{\theta}}`$, constructed as
  $`\mathbb{T}_{\boldsymbol{\theta}}
  \mathbb{T}_{\boldsymbol{\theta}}^\mathsf{T}
  +\mathbb{L}^{\mathrm{skw}}_{\boldsymbol{\theta}}`$.

The symmetric part is positive semidefinite, while the skew part contributes
no dissipation. These architectural constraints hold during both training and
inference.

<p align="center">
  <img src="assets/figure_1.png"
       alt="Overview of constitutive-model discovery with hard-constrained non-equilibrium thermodynamics."
       width="100%">
</p>


## Environment setup

```bash
python -m pip install -r requirements.txt
```

## Included experiments

Run all commands from the repository root.

### Inference from supplied checkpoints

```bash
python drucker_prager_inference.py
python isotropic_hardening_inference.py
```

Each script reconstructs the model and normalization from the training data,
loads a checkpoint with `weights_only=True`, solves for the initial elastic
strain, integrates the unseen protocol, and displays a reference-versus-model
plot.

### Training

```bash
python drucker_prager_training.py
python isotropic_hardening_training.py
```

Retraining writes new files named `*_retrained.pt`; the supplied checkpoints
are not overwritten. Training is intentionally configured for the full epoch
counts used by the draft and can take substantial time on CPU.

For a one-update smoke run:

```bash
HARD_THERMO_EPOCHS=2 python *_training.py
```


## Tests

Unit tests to verify the physical and numerical methods encoded by the implementation:

- ICNN and PICNN convexity
- zero energy and thermodynamic forces at the reference state
- positive-semidefinite symmetric transport and non-negative dissipation
- skew-operator power orthogonality
- differentiability of the training integrator
- loading-knot alignment
- strict compatibility with the two canonical checkpoints.

```bash
python -m unittest discover -s tests -v
```

## Repository layout

```text
hard_thermodynamics/     Local neural architectures and integrators
tests/                   Focused scientific unit tests
data/                    Training and inference trajectories
checkpoints/             Canonical and archived model state dictionaries
assets/                  README figure
docs/                    Reproducibility and release documentation
*.py                     Benchmark training and inference entry points
```

See [DATA.md](data/DATA.md) for serialized dataset schemas,
[checkpoints/README.md](checkpoints/README.md) for checkpoint provenance.

## Citation

```bibtex
@article{masi2026learning,
  title   = {Learning inelastic constitutive models from stress--strain data
             under hard thermodynamic constraints},
  author  = {Masi, Filippo},
  journal = {arXiv preprint arXiv:2605.16837},
  year    = {2026},
  doi     = {10.48550/arXiv.2605.16837}
}
```

Machine-readable citation metadata are available in
[`CITATION.cff`](CITATION.cff).

