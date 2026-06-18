# Authors: Ali Siahkoohi, alisk@ucf.edu
# Date: Nov 2025


export _wsave, set_plot_configs

# NOTE: the `_wsave(s, ::PyPlot.Figure)` method is intentionally NOT defined here.
# The `Rosenbrock` dependency already adds an identical method to the shared
# `DrWatson._wsave` generic (Rosenbrock/src/utils/savefig.jl). Defining a second
# same-signature method here collided with it and was forbidden during module
# precompilation ("Method overwriting is not permitted during Module
# precompilation"). We re-export the `_wsave` symbol (imported via
# `import DrWatson: _wsave` in SVGDADMMSampler.jl) and rely on Rosenbrock's method.
# Behavior is unchanged: all project figures are/were rendered at Rosenbrock's
# dpi=250 (the previous dpi=300 method here was shadowed and never active).


function set_plot_configs(; fontsize = 10)
    set_style("whitegrid")
    rc("font", family = "serif", size = fontsize)
    font_prop = matplotlib.font_manager.FontProperties(
        family = "serif",
        style = "normal",
        size = fontsize,
    )
    sfmt = matplotlib.ticker.ScalarFormatter(useMathText = true)
    sfmt.set_powerlimits((0, 0))
    matplotlib.use("Agg")

    return font_prop, sfmt
end

