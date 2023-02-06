# Trace-Based Evaluation

The `@ppl` macro transform a function to a traced function, where all random draws are recorded.
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
function simple(mean::Float64, constraints::Dict, trace::Dict)
    function inner()
        X = let distribution = Normal(mean, 1.),
                # if we have observations use them, else sample from distribution
                value = haskey(constraints, :X) ? constraints[:X] : rand(distribution),
                lp = logpdf(distribution, value)
                # record value and log probability
                trace[:X] = (value, lp)
                # assign value to X
                value
        end
        let distribution = Normal(X, 1.),
            value = haskey(constraints, :Y) ? constraints[:Y] : rand(distribution),
            lp = logpdf(distribution, value)
            trace[:Y] = (value, lp)
            value
        end
        return X
    end

    return inner(), trace # return trace as well
end
```
This can be checked with the `@macroexpand` function.


We can now call the model with observations `simple(0., Dict(:Y=>1))`, which returns value and trace, and perform inference
```julia
traces, retvals, lps = likelihood_weighting(simple, (0.,), Dict(:Y=>1), 1_000_000);
```