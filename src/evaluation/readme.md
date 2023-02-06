# Evaluation-Based

Similar to the trace-based approach, we also transform a function with the `@ppl` macro.
Instead of a predefined tracing behavior, we allow the implementation of samplers, which decide what happens at sample statements at evaluation time.
We use `Gen.jl` sample syntax `X = {addr} ~ Distribution(args...)`.

For example,

```julia
@ppl function simple(mean::Float64)
    X = {:X} ~ Normal(mean, 1.)
    {:Y} ~ Normal(X, 1.)
    return X
end
```

is (essentially) transformed to

```julia
function simple(mean::Float64, sampler::Sampler, constraints::Dict)
    function inner()
        X = let distribution = Normal(mean, 1.),
                # check if we have observation
                obs = haskey(constraints, :X) ? constraints[:X] : nothing
                # let sampler decide what to do
                value = sample(sampler, :X, distribution, obs)
                # assign value to X
                value
        end
        let distribution = Normal(X, 1.),
            obs = haskey(constraints, :Y) ? constraints[:Y] : nothing
            value = sample(sampler, :Y, distribution, obs)
            value
        end
        return X
    end

    return inner()
end
```
This can be checked with the `@macroexpand` function.

Interaction with the model is possible by implementing custom samplers.

The most basic sampler, simple returns the observations or samples if necessary.
```julia
function sample(sampler::Forward, addr::Any, dist::Distribution, obs::Union{Nothing, Real})::Real
    if !isnothing(obs)
        return obs
    end

    return rand(dist)
end
```

We can now call the model with observations `simple(0., Forward(), Dict(:Y=>1))` and perform inference
```julia
traces, retvals, lps = likelihood_weighting(simple, (0.,), Dict(:Y=>1), 1_000_000);
```