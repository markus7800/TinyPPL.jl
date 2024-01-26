
using TinyPPL.Graph

# model = @pgm uninvoked Diamond begin
#     function or(x, y)
#         max(x, y)
#     end
#     function and(x, y)
#         min(x, y)
#     end
#     function diamond(s1)
#         let route ~ Bernoulli(0.5), # Bernoulli(s1 == 1 ? 0.4 : 0.6),
#             s2 = route == 1. ? s1 : false,
#             s3 = route == 1. ? false : s1,
#             drop ~ Bernoulli(0.0001)

#             or(s2, and(s3, 1-drop))
#         end
#     end
#     let net1 ~ Dirac(diamond(1.)),
#         net2 ~ Dirac(diamond(net1)),
#         net3 ~ Dirac(diamond(net2))

#         net3
#     end
# end
function get_model()
    @pgm uninvoked Diamond begin
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
            let net ~ Dirac(diamond(old_net)) # introduces node for each function result
                net
            end
        end
        @iterate(100, func, 1.)
    end
end

function get_model_factor_graph(N)

    variable_nodes = VariableNode[]
    factor_nodes = FactorNode[]

    net = VariableNode(length(variable_nodes), :net)
    net_factor = FactorNode([net], [-Inf, 0.])
    net.support = [0,1]
    push!(variable_nodes, net)
    push!(factor_nodes, net_factor)

    net_table = reshape([0.0, -Inf, 0.0, -Inf, 0.0, 0.0, 0.0, -Inf, -Inf, 0.0, -Inf, 0.0, -Inf, -Inf, -Inf, 0.0], 2, 2, 2, 2)
    for _ in 1:N
        route = VariableNode(length(variable_nodes), :route)
        route.support = [0,1]
        route_factor = FactorNode([route], [log(0.5), log(0.5)])
        push!(variable_nodes, route)
        push!(factor_nodes, route_factor)

        drop = VariableNode(length(variable_nodes), :drop)
        drop_factor = FactorNode([drop], [log(1-0.001), log(0.001)])
        drop.support = [0,1]
        push!(variable_nodes, drop)
        push!(factor_nodes, drop_factor)

        old_net = net
        net = VariableNode(length(variable_nodes), :net)
        net_factor = FactorNode([old_net, route, drop, net], copy(net_table))
        net.support = [0,1]
        push!(variable_nodes, net)
        push!(factor_nodes, net_factor)
    end

    marginal_variables = [net.variable]


    for f in factor_nodes
        for v in f.neighbours
            push!(v.neighbours, f)
        end
    end

    return_variables = [net]
    return_factor = add_return_factor!(factor_nodes, return_variables)

    variable_nodes, factor_nodes, marginal_variables, return_factor
end

function print_reference_solution(N=100)
    repeatf(n, f, x) = n > 1 ? f(repeatf(n-1, f, x)) : f(x)
    R = 0.5
    D = 0.001
    T0 = 1.
    T(t) = t*(R + (1-D)*(1-R))
    println("Reference: P(1)=", repeatf(N, T, T0))
end
