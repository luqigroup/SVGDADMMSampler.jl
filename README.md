# SVGDADMMSampler.jl

Companion code for *"Dual-space posterior sampling for Bayesian inference in constrained inverse problems"*. This package implements Stein Variational Gradient Descent (SVGD) with ADMM constraints for Bayesian inference and sampling.

## Installation

### Prerequisites

Make sure you have `matplotlib` installed in your Python environment for plotting:
```bash
pip install matplotlib
```

Configure PyCall to use your Python installation:
```julia
ENV["PYTHON"] = "/usr/bin/python3"  # Adjust path to your Python
using Pkg
Pkg.build("PyCall")
# Restart Julia
```

### Install the package

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Reproducing the Results

### Rosenbrock Conditional Posterior Sampling

The Rosenbrock example demonstrates conditional posterior inference $p(\mathbf{x} \mid \mathbf{y})$ where the prior follows the Rosenbrock distribution and observations are $\mathbf{y} = \mathbf{x} + \boldsymbol{\eta}$ with Gaussian noise. We compare ADMM-SVGD (which decomposes the posterior via an auxiliary constraint $z = x_1^2$) against standard SVGD (which uses the direct posterior gradient).

Configuration files are in `config/`. Key parameters:

| Parameter | ADMM-SVGD | Standard SVGD |
|-----------|-----------|---------------|
| Particles | 1000 | 1000 |
| Step size ($\eta$) | 0.15 | 0.05 |
| Iterations | 2500 | 5000 |
| Penalty ($\mu$) | 1.0 | -- |

**Step 1: Run ADMM-SVGD conditional sampling.** This generates the test instances (X_fixed, Y_fixed) and runs ADMM-SVGD for all five observations.

```bash
julia --project=. scripts/admm_svgd_conditional_sampling.jl
```

**Step 2: Run standard SVGD conditional sampling.** This loads the same test instances from Step 1 and runs plain SVGD with the direct posterior gradient.

```bash
julia --project=. scripts/svgd_conditional_sampling.jl
```

**Step 3: Generate paper figures.** This produces all comparison figures: prior, data, ADMM-SVGD posteriors, standard SVGD posteriors, combined overlay, convergence diagnostics, and Q-Q plots.

```bash
julia --project=. scripts/admm_svgd_conditional_paper_figures.jl
```

Figures are saved to `plots/` and to the paper figures directory.

### Rosenbrock Unconditional Sampling

Sample from the Rosenbrock distribution using ADMM-SVGD and visualize the results:

```bash
julia --project=. scripts/admm_svgd_sampling.jl
julia --project=. scripts/admm_svgd_visualization.jl
```

Optionally, run pure SVGD (without ADMM) for comparison:

```bash
julia --project=. scripts/pure_svgd_sampling.jl
julia --project=. scripts/pure_svgd_visualization.jl
```

## Project Structure

```
config/                          # Configuration files (JSON)
  admm_svgd_conditional_sampling.json   # ADMM-SVGD parameters
  svgd_conditional_sampling.json        # Standard SVGD parameters
src/
  SVGDADMMSampler.jl             # Main module
  sampling/
    admm_svgd.jl                 # ADMMSVGDSampler, step!, compute_bandwidth, svgd_update!
    sample.jl                    # MCMC sampler (MALA)
scripts/
  admm_svgd_conditional_sampling.jl     # ADMM-SVGD conditional sampling
  svgd_conditional_sampling.jl          # Standard SVGD conditional sampling
  admm_svgd_conditional_paper_figures.jl  # Generate all paper figures
data/                            # Saved results (JLD2, managed by DrWatson)
plots/                           # Generated figures
```

## Author

Ali Siahkoohi (alisk@ucf.edu)

## License

MIT