
# Reads a discrete Bayesian network in BIF format (https://www.bnlearn.com/bnrepository/).
# Supports `variable` blocks with `type discrete` and `probability` blocks with `table` or one line per parent configuration.
# Returns variable and factor nodes of the corresponding factor graph (log probabilities).
function read_bif(pathname::String)
    s = read(pathname, String) # opens and closes file
    s = replace(s, "\r\n" => "\n") # Windows line endings
    # blocks end with a line that only contains }
    blocks = split(s, r"\n[ \t]*}[ \t]*(?=\n|$)")

    variable_nodes = VariableNode[]
    factor_nodes = FactorNode[]
    name_to_variable = Dict{String, VariableNode}()
    name_to_support_map = Dict{String, Dict{String, Int}}()

    for block in blocks
        block = strip(block)
        isempty(block) && continue
        if startswith(block, "network")
            continue

        elseif startswith(block, "variable")
            # variable A {
            #     type discrete [ 3 ] { young, adult, old };
            # }
            components = split(block)
            # println(components)
            name = String(components[2])
            @assert components[4] == "type"
            @assert components[5] == "discrete"
            i = 6
            while components[i] != "{"
                i += 1
            end
            i += 1
            values = String[] # ["young", "adult", "old"]
            while components[i] != "};"
                push!(values, rstrip(components[i], ','))
                i += 1
            end
            node = VariableNode(length(variable_nodes)+1, name)
            node.support = 1:length(values)
            push!(variable_nodes, node)
            name_to_variable[name] = node
            name_to_support_map[name] = Dict{String, Int}(value => i for (i, value) in enumerate(values))

        elseif startswith(block, "probability")
            lines = [strip(line) for line in split(block, '\n') if !isempty(strip(line))]
            header = lines[1] # probability ( A ) {     or      probability ( E | A, S ) {
            components = split(header)
            @assert components[2] == "("
            name = String(components[3])
            node = name_to_variable[name]
            if components[4] == ")"
                # probability ( A ) {
                #     table 0.3, 0.5, 0.2;
                # }
                # unconditional
                @assert length(lines) == 2 lines
                table = zeros(length(node.support))
                table_line = split(lines[2]) # ["table", "0.3,", "0.5,", "0.2;"]
                @assert table_line[1] == "table"
                for i in 1:length(node.support)
                    table[i] = log(parse(Float64, rstrip(table_line[i+1], [',',';'])))
                end
                factor_node = FactorNode([node], table)
                push!(factor_nodes, factor_node)
            elseif components[4] == "|"
                # probability ( T | O, R ) {
                #     (emp, small) 0.48, 0.42, 0.10;
                #     (self, small) 0.56, 0.36, 0.08;
                #     (emp, big) 0.58, 0.24, 0.18;
                #     (self, big) 0.70, 0.21, 0.09;
                # }
                i = 5
                conditionals = VariableNode[] # name_to_variable.(["O", "R"])
                while components[i] != ")"
                    push!(conditionals, name_to_variable[rstrip(components[i], ',')])
                    i += 1
                end
                table = zeros(length(node.support), [length(c.support) for c in conditionals]...)
                @assert prod(length(c.support) for c in conditionals) == length(lines)-1

                for line in lines[2:end]
                    l = lstrip(line, [' ', '(']) # emp, small) 0.48, 0.42, 0.10;
                    named_value_str, value_str = split(l, ")", limit=2) # ["emp, small", " 0.48, 0.42, 0.10;"]

                    ixs = Int[]
                    for (i,n) in enumerate(split(named_value_str))
                        n = rstrip(n, ',')
                        support_map = name_to_support_map[conditionals[i].address] # "emp" => 1, "small" => 1 etc
                        push!(ixs, support_map[n])
                    end
                    values = log.([parse(Float64, rstrip(v, [',',';'])) for v in split(value_str)])
                    table[:, ixs...] = values
                end
                factor_node = FactorNode(append!([node], conditionals), table) # will sort
                push!(factor_nodes, factor_node)
            else
                error("Cannot parse probability block header: $header")
            end
        else
            error("Unknown block in BIF file: $(first(split(block, '\n')))")
        end
    end
    @assert !isempty(variable_nodes) "No variables found in $pathname."

    for f in factor_nodes
        for v in f.neighbours
            push!(v.neighbours, f)
        end
    end

    return variable_nodes, factor_nodes
end

export read_bif

# value names written to BIF file
bif_value_name(x::Real) = isinteger(x) ? string(Int(x)) : string(x)

# returns all neighbours v of factor f, such that f is a conditional probability table of v given the other neighbours,
# i.e. ∑_v exp(f.table) = 1 for every configuration of the other neighbours.
function get_cpd_child_candidates(f::FactorNode; atol=1e-8)
    candidates = VariableNode[]
    for (i, v) in enumerate(f.neighbours)
        if all(x -> isapprox(x, 0., atol=atol), logsumexp(f.table, i))
            push!(candidates, v)
        end
    end
    return candidates
end

# Writes factor graph to BIF format.
# Every factor has to be a conditional probability table p(child | parents) (log probabilities) and
# every variable has to be the child of exactly one factor.
# The child of each factor is inferred from the table (the neighbour over which the table is normalised).
# Variables are named X<variable> and values are named by their support values.
function write_bif(path::String, variable_nodes::Vector{VariableNode}, factor_nodes::Vector{FactorNode})
    # assign each factor a child
    factor_to_child = Dict{FactorNode, VariableNode}()
    assigned = Set{VariableNode}()
    for f in factor_nodes
        candidates = get_cpd_child_candidates(f)
        unassigned = [v for v in candidates if !(v in assigned)]
        if isempty(unassigned)
            error("Factor over $(f.neighbours) is not a conditional probability table of an unassigned variable, cannot write to BIF.")
        end
        child = unassigned[end]
        factor_to_child[f] = child
        push!(assigned, child)
    end
    for v in variable_nodes
        @assert v in assigned "Variable $v has no conditional probability table."
    end

    open(path, "w") do io
        println(io, "network unknown {\n}")
        for v in variable_nodes
            print(io, "variable X", v.variable)
            println(io, " {")
            print(io, "  type discrete [ $(length(v.support)) ] { ")
            print(io, join(bif_value_name.(v.support), ", "))
            println(io, " };")
            println(io, "}")
        end
        for f in factor_nodes
            child = factor_to_child[f]
            child_ix = findfirst(==(child), f.neighbours)
            parent_ixs = [i for i in eachindex(f.neighbours) if i != child_ix]
            parents = f.neighbours[parent_ixs]

            print(io, "probability ( X", child.variable)
            if !isempty(parents)
                print(io, " | ")
                print(io, join(map(x -> "X$(x.variable)", parents), ", "))
            end
            println(io, " ) {")
            if isempty(parents)
                print(io, "  table ")
                print(io, join(exp.(f.table), ", "))
                println(io, ";")
            else
                # one line per parent configuration
                for parent_values in Iterators.product([1:length(p.support) for p in parents]...)
                    index = Vector{Any}(undef, length(f.neighbours))
                    index[parent_ixs] .= parent_values
                    index[child_ix] = Colon()
                    print(io, "  (")
                    print(io, join([bif_value_name(p.support[i]) for (p, i) in zip(parents, parent_values)], ", "))
                    print(io, ") ")
                    print(io, join(exp.(f.table[index...]), ", "))
                    println(io, ";")
                end
            end
            println(io, "}")
        end
    end
end

export write_bif
