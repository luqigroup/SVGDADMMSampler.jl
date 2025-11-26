# SVGDADMMSampler.jl

A Julia package implementing Stein Variational Gradient Descent (SVGD)
with ADMM constraints for Bayesian inference and sampling. The package
also includes normalizing flow and Lagevin dynamics baselines for comparison.

## Sampling Methods

#### ADMM-SVGD (`src/sampling/admm_svgd.jl`)

The core algorithm combining Stein Variational Gradient Descent with the
Alternating Direction Method of Multipliers for constrained posterior
sampling. The method maintains a set of particles that collectively
approximate the target posterior while satisfying observational
constraints through the ADMM framework.

#### pSGLD (`src/sampling/pSGLD.jl`)

Preconditioned Stochastic Gradient Langevin Dynamics for MCMC-based sampling, used as a baseline for comparison with SVGD methods.

### Normalizing Flow Samplers

The package includes conditional and unconditional normalizing flow models (Glow architecture) for comparison with SVGD methods:
- **NetworkGlow**: Unconditional generative model trained via maximum likelihood
- **NetworkConditionalGlow**: Conditional generative model for amortized posterior inference

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

## Scripts

### SVGD-Based Scripts

#### ADMM-SVGD Sampling
- **`admm_svgd_sampling.jl`**: Runs ADMM-SVGD to sample from constrained posteriors, combining prior knowledge with data-fidelity constraints
- **`admm_svgd_conditional_sampling.jl`**: Performs conditional ADMM-SVGD sampling for multiple test observations
- **`admm_svgd_visualization.jl`**: Visualizes ADMM-SVGD sampling results including particle evolution and constraint satisfaction
- **`admm_svgd_conditional_visualization.jl`**: Generates visualizations for conditional ADMM-SVGD experiments

#### Pure SVGD
- **`pure_svgd_sampling.jl`**: Runs standard SVGD without ADMM constraints for unconditional prior sampling
- **`pure_svgd_visualization.jl`**: Creates visualizations for pure SVGD results, including convergence diagnostics

### Normalizing Flow Scripts

#### Training
- **`rosenbrock_training.jl`**: Trains an unconditional Glow model on Rosenbrock distribution via maximum likelihood
- **`rosenbrock_amortized_training.jl`**: Trains a conditional Glow model for amortized inference given observations

#### Sampling
- **`rosenbrock_sampling.jl`**: Samples from trained unconditional Glow model
- **`rosenbrock_amortized_sampling.jl`**: Performs conditional sampling from trained amortized Glow model and compares with MCMC baselines

## Running Examples

### SVGD-Based Sampling

Run ADMM-SVGD for constrained sampling:

```julia
julia --project=. scripts/admm_svgd_sampling.jl
julia --project=. scripts/admm_svgd_visualization.jl
```

Run pure SVGD for unconditional sampling:

```julia
julia --project=. scripts/pure_svgd_sampling.jl
julia --project=. scripts/pure_svgd_visualization.jl
```

### Normalizing Flow Baselines

Train and sample from flow models:
```julia
# Unconditional model
julia --project=. scripts/rosenbrock_training.jl
julia --project=. scripts/rosenbrock_sampling.jl

# Conditional (amortized) model
julia --project=. scripts/rosenbrock_amortized_training.jl
julia --project=. scripts/rosenbrock_amortized_sampling.jl
```

## Author

Ali Siahkoohi (alisk@ucf.edu)

## License

MIT