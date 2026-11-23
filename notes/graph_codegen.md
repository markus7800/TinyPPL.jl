# Code generation for graph models (`@pgm`)

Design note on how `src/graph` turns a symbolic PGM into Julia functions, why it uses
RuntimeGeneratedFunctions.jl (RGF), and what the performance trade-offs are (REVIEW.md J2, J4).

## What is generated

For each model, `compile_symbolic_pgm` (src/graph/core/core.jl) generates:

- one distribution function per variable (`pgm.distributions[i](X)`) and the return expression,
- `sample!`, `logpdf`, `unconstrained_logpdf`, `transform_to_constrained!`, `transform_to_unconstrained!`,
  which are fully unrolled over the variables in topological order,
- for `plated` models, one function per plate and kind (`:lp`, `:sample`, ...), called from the functions above.

`compile_likelihood_weighting` and `compile_lmh` / `compile_rwmh` (src/graph/compiled) generate further
unrolled functions from the same symbolic distributions.

## Before: `eval` of globally named functions

Originally every function was `eval`ed as a global function in `TinyPPL.Graph`, named after the model
(`<name>_dist_<i>`, `<name>_logpdf`, `<name>_lp_plate_<symbol>`, ...).
The intention was performance: a call to a global named function from other generated code
(e.g. logpdf → plate function) is a static call that can be inlined, with no indirection through a function pointer.

Problems:

- **Name collisions:** compiling a model with the same name again redefined the methods, so existing PGMs
  silently used the new model's functions. Plated models were affected even through the generated logpdf,
  which calls the plate functions by name. `compile_lmh` and `compile_rwmh` both named their kernels
  `<name>_lmh_<node>`, so compiling one overwrote the kernels of the other.
- **World age:** `m = @pgm ...; m.logpdf(...)` inside one function failed with "method too new",
  because the methods were defined after the calling function started running (`invokelatest` was needed internally).
- **Scope:** static parts of the program and `$(...)` interpolations were evaluated in `TinyPPL.Graph`,
  so data had to be referenced as `$(Main.x)`.

## Now: RuntimeGeneratedFunctions

All generated functions are created with `make_function` (core.jl), i.e.
`RuntimeGeneratedFunction(TinyPPL.Graph, TinyPPL.Graph, expr)`:

- Functions are values without global names → no collisions. Identical expressions share code (cached by hash).
- An RGF can be called in the world in which it was created → no world age errors, no `invokelatest`.
- Each RGF has its own concrete type (the expression hash is a type parameter), just like the singleton type
  of a named function. Generated code calls other generated functions by interpolating the function value
  (`:($plate_f($X, $Y))`, see `plate_function` in plates.jl), which is a static call and gets inlined.
- Global names in the function bodies are still resolved in `TinyPPL.Graph` (option A below).

Scoping of `@pgm` (`pgm_macro`):

- addresses, loop ranges and observations are evaluated in the module where `@pgm` is called (its global variables are visible),
- `$(...)` is evaluated where `@pgm` is called, local variables included (the interpolated expressions are escaped).

Caveats:

- RGF ignores argument type annotations (`X::INPUT_VECTOR_TYPE` is documentation only).
- A one-argument anonymous function needs a trailing comma: `function ($X::INPUT_VECTOR_TYPE,) ... end`.
- Stack traces show `RuntimeGeneratedFunction{...hash...}` instead of a model-specific name; printing an RGF shows its body.
- RGF 0.5.12 is used (works with Julia 1.9; newer versions need newer Julia).

### Options considered for the context module

- **A (implemented):** context `TinyPPL.Graph`. Small change, fixes collisions and world age;
  user data and functions enter the model via `$(...)` or as globals in static parts.
- **B:** context = calling module, so user functions resolve directly in distribution expressions.
  Every internal name in generated code (`INPUT_VECTOR_TYPE`, `Flat`, `logpdf`, transforms, ...) would have to be
  fully qualified, and `RuntimeGeneratedFunctions.init` must run on the calling module at macro expansion time
  (lazily at runtime hits world age again; tested to work from the macro).

## Measurements (scratch benchmarks, Julia 1.9.1)

Outer generated function calling an inner one 3 times:

| inner called | time | `@code_typed` |
|---|---|---|
| by global name (before) | 1.67 ns | inner inlined |
| as interpolated RGF value (now) | 1.67 ns | identical code, inner inlined |

100 calls of a model function through a struct field, as inference algorithms do:

| field type | named function | RGF |
|---|---|---|
| `f::Function` (current `PGM`) | 2.03 μs, 200 allocations | 2.07 μs, 200 allocations |
| `f::F` (type parameter) | 36 ns, 0 allocations | 36 ns, 0 allocations |

So RGF costs nothing inside generated code. The real indirection is the abstractly typed fields of `PGM`
(`logpdf::Function`, `distributions::Vector{Function}`): every call from an inference loop is a dynamic dispatch,
and the `Float64` result is boxed. Global names did not help there.

## Possible next steps for performance

- **Concretely typed PGM:** `PGM{F_sample, F_logpdf, ...}` with function-typed fields. Removes dynamic dispatch
  at the boundary between inference algorithm and model (about 55× in the micro benchmark above).
  Cost: every inference algorithm is compiled again per model (compile latency).
  `distributions` could become a tuple, but indexing it with a runtime node index is still type unstable.
- **More unrolling / per-node functions:** code that indexes `distributions[node]` at runtime
  (`get_distribution` in lmh/rwmh, LW, SMC, exact inference setup) still dispatches dynamically, and the distribution
  types differ by node anyway. The compiled kernels (`compile_lmh`, `compile_likelihood_weighting`) show the way:
  generate one function per node that does the whole job (e.g. score of all children), so no distribution object
  crosses a function boundary. Fully unrolling the topological order everywhere would follow the same pattern.
