## Handler-Based

The implementation is adapted from [MiniPyro](https://pyro.ai/examples/minipyro.html).
For a detailed explanation read tutorial [tutorial](https://pyro.ai/examples/effect_handlers.html).

In summary, we have normal JULIA functions (no meta-programming), which invoke
```julia
sample(name::Any, dist::Distribution, args...; obs=nothing)
```

For example,
```julia
function simple(mean, y)
    X = sample(:X, Normal(mean, 1.))
    sample(:Y, Normal(X, 1.), obs=y)
    return X
end
```

We can wrap the function with handlers, for instance
```julia
trace_handler = trace(simple)
```
When we call a handler, it is registered on the `HANDLER_STACK` and its parent handlers are called recursively. In the above case, the parent handler is the model itself, which sends a message to all the handlers in the stack, when it encounters a `sample` statement.
The `trace_handler` will record the values of the random choices.

```julia
trace_handler(0., 1.) # arguments are forwarded to model
model_trace = trace_handler.trace
```

We can chain handlers:
```julia
replay_handler = trace(
    block(
        replay(simple, model_trace),
        msg -> msg["type"] == "observation"
    )
)
```

When we call `replay_handler(0., 1.)`, first `simple` sends a message for each sample statement to the `replay` handler, which adds the corresponding value from `mode_trace` to the message and passes the message to the `block` handler.
This handler stops messages that do not fulfill the filtering criterion, i.e. it only forwards messages that correspond to sample statements with observed values.
Lastly, the `trace` handler will record all remaining messages.
Then the handlers get a chance to post-process the message in reversed order.

Therefore,
```julia
replay_handler.trace
```
contains all observed values from `model_trace`.

(This chain of handlers is described for demonstration purposes, we could have simple filtered the `model_trace` directly).