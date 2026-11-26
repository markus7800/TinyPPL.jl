# Exact inference on Bayesian networks from the bnlearn repository (https://www.bnlearn.com/bnrepository/),
# read from BIF files with read_bif, which returns the factor graph (variable nodes and factor nodes) directly.
#
# - survey (6 variables): marginals of all variables, variable elimination vs. a calibrated junction tree.
# - munin (1041 variables): a calibrated junction tree answers many queries faster than running variable elimination
#   for each query.
# - Exporting a network as model code for other probabilistic programming languages.
#
# The networks are downloaded to exact_inference/bif/ on first use (needs `gunzip`).
# Run: julia --project=examples examples/exact_inference/bif_networks.jl

using TinyPPL.Graph
import Downloads
import Random

function bif_path(name::String)
    dir = joinpath(@__DIR__, "bif")
    path = joinpath(dir, name * ".bif")
    if !isfile(path)
        mkpath(dir)
        url = "https://www.bnlearn.com/bnrepository/$name/$name.bif.gz"
        println("Downloading ", url)
        Downloads.download(url, path * ".gz")
        run(`gunzip $(path * ".gz")`)
    end
    return path
end

# ===== Marginals of a small network =====
variable_nodes, factor_nodes = read_bif(bif_path("survey"))
# A calibrated junction tree gives the marginals of all variables at once.
order = get_greedy_elimination_order(variable_nodes, Int[])
junction_tree, root, root_factor = get_junction_tree(variable_nodes, order, factor_nodes[1])
result = junction_tree_message_passing(junction_tree, root, root_factor, true)
marginals = Dict(v.address => p for (v, p) in get_marginals(result))
println("survey network:")
for v in variable_nodes
    ve = greedy_variable_elimination(variable_nodes, factor_nodes, [v]) # one run per variable
    println("  P(", v.address, ") = ", round.(marginals[v.address], digits=4), ", equals VE: ", marginals[v.address] ≈ exp_normalised_table(ve.factor))
end


# ===== Many queries on a large network =====
variable_nodes, factor_nodes = read_bif(bif_path("munin"))
println("\nmunin network: ", length(variable_nodes), " variables")

t = @elapsed begin
    order = get_greedy_elimination_order(variable_nodes, Int[])
    junction_tree, root, root_factor = get_junction_tree(variable_nodes, order, factor_nodes[1])
    # with_division computes the messages of the backward pass by dividing out the message of the forward pass
    result = junction_tree_message_passing(junction_tree, root, root_factor, true; with_division=true)
end
println("  building and calibrating the junction tree (", length(junction_tree), " clusters): ", round(t, digits=1), "s")

# Joint posterior of random pairs of variables.
Random.seed!(0)
t_ve, t_jt, all_equal = 0., 0., true
for _ in 1:10
    v1, v2 = rand(variable_nodes, 2)
    v1 == v2 && continue
    global t_ve += @elapsed ve = greedy_variable_elimination(variable_nodes, factor_nodes, [v1, v2])
    global t_jt += @elapsed q = query(result, [v1.variable, v2.variable])
    global all_equal &= isapprox(exp_normalised_table(ve.factor), exp_normalised_table(q.factor), rtol=1e-5)
end
println("  10 queries: VE ", round(t_ve, digits=1), "s, calibrated junction tree ", round(t_jt, digits=1), "s, same results: ", all_equal)


# ===== Exporting a network as model code =====
# Prints the conditional probability tables and the sample statements of a network in topological order,
# in the syntax of several probabilistic programming languages, e.g. to compare them on the same network.

# Each factor of a BIF network is the conditional probability table (CPT) of one variable given its parents.
# read_bif sorts the variables of each factor by index (as required by factor operations), but it creates the factors
# in the order of the `probability ( X | parents ) { ... }` blocks of the file, so we read X from these headers.
struct CPT
    variable::VariableNode
    parents::Vector{VariableNode}
    table::Array{Float64} # probabilities indexed by [value, parent values...]
end

function get_cpts(path::String)
    variable_nodes, factor_nodes = read_bif(path)
    names = [m.captures[1] for m in eachmatch(r"probability\s*\(\s*([^\s|)]+)", read(path, String))]
    @assert length(names) == length(factor_nodes)
    cpts = CPT[]
    for (name, f) in zip(names, factor_nodes)
        i = findfirst(v -> v.address == name, f.neighbours)
        perm = [i; setdiff(eachindex(f.neighbours), i)]
        push!(cpts, CPT(f.neighbours[i], f.neighbours[perm[2:end]], round.(exp.(permutedims(f.table, perm)), digits=10)))
    end
    return cpts
end

# Parents before children.
function topological_sort(cpts::Vector{CPT})
    ordered = CPT[]
    done = Set{VariableNode}()
    while length(ordered) < length(cpts)
        ready = [cpt for cpt in cpts if !(cpt.variable in done) && all(p in done for p in cpt.parents)]
        @assert !isempty(ready) "the network has a cycle"
        for cpt in ready
            push!(ordered, cpt)
            push!(done, cpt.variable)
        end
    end
    return ordered
end

# Nested lists for Python / JavaScript: the outermost index is the first parent, the innermost the variable.
function nested_list(a::AbstractArray)
    ndims(a) == 1 && return "[" * join(a, ", ") * "]"
    return "[" * join([nested_list(selectdim(a, 2, i)) for i in axes(a, 2)], ", ") * "]"
end

function print_model_code(path::String)
    cpts = topological_sort(get_cpts(path))
    name(cpt) = cpt.variable.address
    cpt_args(cpt) = join(vcat(["\"$(name(cpt))\""], [p.address for p in cpt.parents]), ", ")

    println("# Julia: conditional probability tables (probabilities, indexed by [value, parent values...])")
    println("CPTs = Dict{String, Array{Float64}}(")
    for cpt in cpts
        println("    \"", name(cpt), "\" => ", cpt.table, ",")
    end
    println(")")
    println("\n# Julia, with an explicit context:")
    for cpt in cpts
        println("    $(name(cpt))::Int = sample(ctx, \"$(name(cpt))\", Categorical(get_cpt(CPTs, $(cpt_args(cpt)))), check_args=false)")
    end
    println("\n# with ~ statements:")
    for cpt in cpts
        println("    $(name(cpt)) ~ categorical(get_cpt(CPTs, $(cpt_args(cpt))))")
    end
    println("\n# addresses: ", join([":" * name(cpt) for cpt in cpts], ", "))
    println("\n# Python / JavaScript: conditional probability tables")
    println("CPTs = {")
    for cpt in cpts
        println("    \"", name(cpt), "\": ", nested_list(cpt.table), ",")
    end
    println("}")
    println("\n# JavaScript:")
    for cpt in cpts
        println("    var $(name(cpt)) = discrete(get_cpt_$(length(cpt.parents))($(cpt_args(cpt))))")
    end
    println("\n# reading values from a context:")
    for cpt in cpts
        println("    $(name(cpt)) = manual_read(ctx, \"$(name(cpt))\")")
    end
end

println()
print_model_code(bif_path("survey"))
