
using TinyPPL.Graph

function to_net_file(path, variable_nodes, factor_nodes)
    open(path, "w") do io
        println(io, "net\n{\n}")
        for v in variable_nodes
            println(io, "node X", v.variable)
            println(io, "{")
            print(io, "  states = ( ")
            for s in v.support
                print(io, "\"", Int(s), "\" ")
            end
            println(io, ");")
            println(io, "}")
        end
        function print_prob_recurse(vars::Vector{VariableNode}, f::FactorNode, index=Int[])
            if isempty(vars)
                @assert !isempty(index)
                print(io, "(")
                print(io, join(f.table[index..., :], " "))
                print(io, ")")
            else
                v = popfirst!(vars)
                print(io, "(")
                for i in 1:length(v.support)
                    push!(index, i)
                    print_prob_recurse(vars, f, index)
                    pop!(index)
                end
                print(io, ")")
                pushfirst!(vars, v)
            end
        end
        for v in factor_nodes
            print(io, "potential ( X", v.neighbours[end].variable)
            if length(v.neighbours) > 1
                print(io, " | ")
                for n in v.neighbours[1:end-1]
                    print(io, "X", n.variable, " ")
                end
                println(io, ")")
            else
                println(io, " )")
            end
            println(io, "{")
            print(io, "  data = ")
            if length(v.neighbours) > 1
                print_prob_recurse(v.neighbours[1:end-1], v)
                println(io, ";")
            else
                print(io, "(")
                print(io, join(v.table, " "))
                println(io, ");")
            end
            println(io, "}")
        end
    end
end

function to_bif_file(path, variable_nodes, factor_nodes)
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


N = 5000
model = @ppl uninvoked Diamond begin
    function or(x, y)
        max(x, y)
    end
    function and(x, y)
        min(x, y)
    end
    function diamond(s1)
        let route ~ Bernoulli(0.5), # Bernoulli(s1 == 1 ? 0.4 : 0.6),
            s2 = route == 1. ? s1 : false,
            s3 = route == 1. ? false : s1,
            drop ~ Bernoulli(0.001)

            or(s2, and(s3, 1-drop))
        end
    end
    function func(old_net)
        let net ~ Dirac(diamond(old_net))
            net
        end
    end
    @iterate(5000, func, 1.)
end;

println("Get factor graph:")
variable_nodes, factor_nodes = get_factor_graph(model, logscale=false, sorted=false);
to_net_file("/Users/markus/Documents/AQUA/ppl_to_pgm/diamond_$N.net", variable_nodes, factor_nodes)
to_bif_file("/Users/markus/Documents/AQUA/ppl_to_pgm/diamond_$N.bif", variable_nodes, factor_nodes)

open("/Users/markus/Documents/AQUA/ppl_to_pgm/diamond_$(N)_return_variables.txt", "w") do io
    println(io, ["X$v" for v in return_expr_variables(model)])
end

marginal_variables = return_expr_variables(model)
elimination_order = get_elimination_order(model, variable_nodes, marginal_variables, :Topological);
open("/Users/markus/Documents/AQUA/ppl_to_pgm/diamond_$(N)_elimination_order.txt", "w") do io
    println(io, ["X$(v.variable)" for v in elimination_order])
end