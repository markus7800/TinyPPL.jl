# Examples

## [Geometric](geometric/)
```
i ~ Geometric(0.5)
X ~ Normal(i, 1.0)
infer p(i | X=5.)
```
Geometric is implemented as a while loop (or recursively) of successive Bernoulli flips (unbounded number of random variables / higher order).

## [Linear Regression](linear_regression/)
```
slope ~ Normal(0., 10.)
intercept ~ Normal(0., 10.)
y[i] ~ Normal(slope * x[i] + intercept, 1.) for i in 1...100
infer p(slope, intercept | x, y)
```