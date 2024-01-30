
function read_bif(pathname::String)
    f = open(pathname, "r")
    s = read(f, String)
    blocks = split(s, "}\n")

    variable_nodes = VariableNode[]
    factor_nodes = FactorNode[]
    name_to_variable = Dict{String, VariableNode}()
    name_to_support_map = Dict{String, Dict{String, Int}}()

    for block in blocks
        if startswith(block, "variable")
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
            lines = split(block, '\n')
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
                @assert length(lines) == 3
                table = zeros(length(node.support))
                table_line = split(lines[2]) # ["table", "0.3,", "0.5,", "0.2;"]
                for i in 1:length(node.support)
                    table[i] = log(parse(Float64, rstrip(table_line[i+1], [',',';'])))
                end
                factor_node = FactorNode([node], table)
                push!(factor_nodes, factor_node)
            else components[4] == "|"
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
                @assert prod(length(c.support) for c in conditionals) == length(lines)-2
                 
                for line in lines[2:end-1]
                    l = lstrip(line, [' ', '(']) # emp, small) 0.48, 0.42, 0.10;
                    named_value_str, value_str = split(l, ") ", limit=2) # ["emp, small", "0.48, 0.42, 0.10;"]

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
            end
        end
    end
    for f in factor_nodes
        for v in f.neighbours
            push!(v.neighbours, f)
        end
    end

    return variable_nodes, factor_nodes
end

export read_bif

function write_bif(path::String, variable_nodes::Vector{VariableNode}, factor_nodes::Vector{FactorNode})
    open(path, "w") do io
        println(io, "network unknown {\n}")
        for v in variable_nodes
            print(io, "variable X", v.variable)
            println(io, " {")
            print(io, "  type discrete [ $(length(v.support)) ] { ")
            print(io, join(Int.(v.support), ", "))
            println(io, " };")
            println(io, "}")
        end
        function print_prob_recurse(vars::Vector{VariableNode}, f::FactorNode, index=Int[])
            if isempty(vars)
                @assert !isempty(index)
                @assert length(index) == length(f.neighbours)-1
                print(io, "  (")
                values = map(t -> Int(t[2].support[index[t[1]]]), enumerate(f.neighbours[1:end-1]))
                print(io, join(values, ", "))
                print(io, ") ")
                print(io, join(f.table[index..., :], ", "))
                println(io, ";")
            else
                v = popfirst!(vars)
                for i in 1:length(v.support)
                    push!(index, i)
                    print_prob_recurse(vars, f, index)
                    pop!(index)
                end
                pushfirst!(vars, v)
            end
        end
        for v in factor_nodes
            print(io, "probability ( X", v.neighbours[end].variable)
            if length(v.neighbours) > 1
                print(io, " | ")
                print(io, join(map(x -> "X$(x.variable)", v.neighbours[1:end-1]), ", "))
                print(io, " )")
            else
                print(io, " )")
            end
            println(io, " {")
            if length(v.neighbours) > 1
                print_prob_recurse(v.neighbours[1:end-1], v)
            else
                print(io, "  table ")
                print(io, join(v.table, ", "))
                println(io, ";")
            end
            println(io, "}")
        end
    end
end

export write_bif