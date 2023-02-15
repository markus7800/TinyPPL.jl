# Graph-Based

For a graph based approach we need to restrict the source language.
For instance, to avoid an unbounded number of nodes we cannot allow dynamic loops, or recursion.

We can write let blocks

```julia
let var1 = val1,
    var2 ~ distribution(args...)

    body

end
```
vectors and tuples
```julia
[expr1, expr2, ...]
(expr1, expr2, ...)
```
static generators
```julia
[expr for ix in range]
```
where we have to be able to statically evaluate `range`,
and first-order functions without recursion
```julia
function name(args...)
    body
end
```

Sample statements have syntax
```
var ~ distribution(args...)
```
and observe statements (which return `value` when evaluated)
```
distribution(args...) ↦ value
```
Data has to be inlined.
All functions have to be defined in the model block.
Only base JULIA functions and distributions are available.


For example,

```julia
model = @ppl simple begin
    let X ~ Normal(0., 1.)
        Normal(X, 1.) ↦ 1.
        X
    end
end
```

returns a probabilistic graphical model with structure
```
Variables:
x1 ~ Normal(0.0, 1.0)
y2 ~ Normal(x1, 1.0)
Observed:
y2 = 1.0
Dependencies:
x1 → y2
Return expression:
x1
```

A linear regression model can be written as
``` julia
model = @ppl LinReg begin
    function f(slope, intercept, x)
        intercept + slope * x
    end
    let xs = [1., 2., 3., 4., 5.],
        ys = [2.1, 3.9, 5.3, 7.7, 10.2],
        slope ~ Normal(0.0, 10.),
        intercept ~ Normal(0.0, 10.)

        [(Normal(f(slope, intercept, xs[i]), 1.) ↦ ys[i]) for i in 1:5]

        (slope, intercept)
    end
end
```

with structure

```
x1 ~ Normal(0.0, 10.0)
x5 ~ Normal(0.0, 10.0)
y2 ~ Normal(x5 + x1 * 5.0, 1.0)
y3 ~ Normal(x5 + x1 * 2.0, 1.0)
y4 ~ Normal(x5 + x1 * 3.0, 1.0)
y6 ~ Normal(x5 + x1 * 1.0, 1.0)
y7 ~ Normal(x5 + x1 * 4.0, 1.0)
Observed:
y2 = 10.2
y3 = 3.9
y4 = 5.3
y6 = 2.1
y7 = 7.7
Dependencies:
x1 → y2
x1 → y3
x1 → y4
x1 → y6
x1 → y7
x5 → y2
x5 → y3
x5 → y4
x5 → y6
x5 → y7
Return expression:
(x1, x5)
```

In the graph-based approach have great control over the model and can sample at each node with
`rand(model.distributions[node](X))`, where `X` is a vector where `X[i]` corresponds to the `i`-th variable (i.e. `xi` or `yi`). The result is stored in `X[node]`.

Note that the function `model.distributions[node]` returns a distribution which depends only on the parents of `node`.
We can sample from the complete model by calling `rand(model.distributions[node](X))` in topological order.

## Inference

```julia
traces, retvals, lps = likelihood_weighting(model, 1_000_000);
```

TODO: x[1] = 1 not allowed, immutable, but y = x[1] is allowed