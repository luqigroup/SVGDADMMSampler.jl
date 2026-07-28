export load_experiment

"""
   load_experiment(sim_name::String)

   A function that uses the input variables to the experiment, passed by dictionaty
   `parsed_args`, generated via `ArgParse.jl` package, to determine which experiment data to
   load from `datadir(sim_name)`, where `sim_name` is the simulation name.

   In case there are multiple experiments with the same variable valus, the last experiment
   will be returned

    Args:
        parsed_args: A dictionary of input variables.
        return_path: A boolian indicating whether or not return the absolute path to the
            experiment.


"""

function load_experiment(
    parsed_args::Dict,
    keys_to_load::Vector{String};
    return_path = false,
    train = false,
)
    loaded_keys = Dict()
    sim_name = parsed_args["sim_name"]
    !isdir(datadir(sim_name)) && mkpath(datadir(sim_name))
    experiments = collect_results(
        datadir(sim_name),
        black_list = ["Params", "fval", "fval_eval", "opt"],
    )

    # Drop experiments that miss one or some of the input arguments
    if size(experiments, 1) > 0
        # dropmissing!(experiments)
        for (key, value) in parsed_args
            dropmissing!(experiments, key)
        end

        # Keep experiment with same variables
        for (key, value) in parsed_args
            if !train || key != "epoch"
                experiments = experiments[experiments[!, string(key)].==value, :]
            end
        end
    end

    if size(experiments, 1) < 1 && train
        @info "No saved experiments found"
        for key in keys_to_load
            loaded_keys[key] = nothing
        end
        return loaded_keys
    elseif size(experiments, 1) < 1 && !train
        @error "No saved experiments found with such input values"
    end

    @info (string(size(experiments, 1)) * " experiment(s) found — loading the most recent")
    if hasproperty(experiments, :epoch)
        experiment = experiments[partialsortperm(experiments.epoch, 1, rev = true), :]
        experiment_path = experiment[:path]
    else
        experiment_path = experiments[1, :path]
    end
    for key in keys_to_load
        loaded_keys[key] = wload(experiment_path)[key]
    end

    if return_path
        return loaded_keys, experiment_path
    else
        return loaded_keys
    end
end
