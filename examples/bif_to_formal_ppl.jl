using TinyPPL.Graph


function read_bif_2(pathname::String)
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
                    table[i] = (parse(Float64, rstrip(table_line[i+1], [',',';'])))
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
                    values = ([parse(Float64, rstrip(v, [',',';'])) for v in split(value_str)])
                    table[:, ixs...] = values
                end
                factor_node = FactorNode(append!([node], conditionals), table, disable_sort=true)
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

variable_nodes, factor_nodes = read_bif_2("examples/bif_models/alarm.bif");

function topological_sort(variable_nodes::Vector{VariableNode}, factor_nodes::Vector{FactorNode})
    var_to_factor = Dict{VariableNode,FactorNode}(factor_node.neighbours[1] => factor_node for factor_node in factor_nodes)
    for factor in factor_nodes
        println(factor)
    end

    edges = Set{Pair{VariableNode,VariableNode}}()
    for factor_node in factor_nodes
        for neighbour in factor_node.neighbours[2:end]
            push!(edges, neighbour => factor_node.neighbours[1])
        end
    end
    for edge in edges
        println(edge)
    end

    L = VariableNode[]
    S = VariableNode[factor_node.neighbours[1] for factor_node in factor_nodes if length(factor_node.neighbours) == 1]
    while length(S) > 0
        n = pop!(S)
        println("Pop ", n)
        push!(L, n)
        children = [y for (x,y) in edges if x == n]
        for m in children
            @assert (n => m) in edges
            delete!(edges, n => m)
            println("delete edge ", n => m)
            if sum(1 for (x,y) in edges if y == m; init=0) == 0
                println("add to S ", m)
                push!(S, m)
            end
        end
    end
    return [var_to_factor[v] for v in L]
end

factor_nodes = topological_sort(variable_nodes, factor_nodes);

begin
    println("CPTs = Dict{String, Array{Float64}}(")
    for f in factor_nodes
        println("    \"", f.neighbours[1].address, "\" => ", f.table, ",")
    end
    println(")")
end

begin
    for f in factor_nodes
        addr = f.neighbours[1].address
        println("    $addr::Int = sample(ctx, \"$addr\", Categorical(get_cpt(CPTs, ", join(vcat(["\"$addr\""], [neighbour.address for neighbour in f.neighbours[2:end]]), ", "), "), check_args=false))")
    end
end

begin
    for f in factor_nodes
        addr = f.neighbours[1].address
        println("    $addr ~ categorical(get_cpt(CPTs, ", join(vcat(["\"$addr\""], [neighbour.address for neighbour in f.neighbours[2:end]]), ", "), "))")
    end
end

begin
    println(join([":" * f.neighbours[1].address for f in factor_nodes], ", "))
end


function print_nested(a::Array)
    s = size(a)
    if length(s) == 1
        print(a)
    else
        print("[")
        selector = Union{Int,Colon}[Colon() for _ in  1:length(s)]
        for i in 1:s[2]
            selector[2] = i
            print_nested(a[selector...])
            i != s[2] && print(", ")
        end
        print("]")
    end
end

begin
    println("CPTs = {")
    for f in factor_nodes
        print("    \"", f.neighbours[1].address, "\": ")
        print_nested(f.table)
        println(",")
    end
    println("}")
end

begin
    for f in factor_nodes
        addr = f.neighbours[1].address
        n_ix = length(f.neighbours)-1
        println("    var $addr = discrete(get_cpt_$n_ix(", join(vcat(["\"$addr\""], [neighbour.address for neighbour in f.neighbours[2:end]]), ", "), "))")
    end
end


begin
    for f in factor_nodes
        addr = f.neighbours[1].address
        println("   $addr = manual_read(ctx, \"$addr\")")
    end
end