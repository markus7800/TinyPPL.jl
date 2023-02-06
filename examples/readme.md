# Examples

## Geometric
```
i ~ Geometric(0.5)
X ~ Normal(i, 1.0)
infer p(i | X=5.)
```
Geometric is implemented as a while loop (or recursively) of successive Bernoulli flips (unbounded number of random variables / higher order).