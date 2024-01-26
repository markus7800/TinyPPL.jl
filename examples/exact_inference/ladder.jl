
using TinyPPL.Graph

function get_model()
    @pgm uninvoked Ladder begin
        function ladder(s1, s2)
            if s1 == 1.
                let d ~ Bernoulli(0.001)
                    if d == 1.
                        (0., 0.)
                    else
                        let f ~ Bernoulli(0.6)
                            (f, 1. - f)
                        end
                    end
                end
            elseif s2 == 1.
                let f ~ Bernoulli(0.6)
                    (f, 1. - f)
                end
            else
                (0., 0.)
            end
        end
        function func(old)
            let x = ladder(old[1], old[2]),
                x1 ~ Dirac(x[1]),
                x2 ~ Dirac(x[2])
                (x1, x2)
            end
        end
        @iterate(2, func, (1., 0.))
    end
end

function get_model_factor_graph(N)

    variable_nodes = VariableNode[]
    factor_nodes = FactorNode[]

    x1 = VariableNode(length(variable_nodes), :x1)
    x1_factor = FactorNode([x1], [-Inf, 0.])
    x1.support = [0,1]
    push!(variable_nodes, x1)
    push!(factor_nodes, x1_factor)

    x2 = VariableNode(length(variable_nodes), :x2)
    x2_factor = FactorNode([x2], [0., -Inf])
    x2.support = [0,1]
    push!(variable_nodes, x2)
    push!(factor_nodes, x2_factor)

    x1_table = reshape([0.0, 0.0, 0.0, 0.0, 0.0, -Inf, 0.0, -Inf, 0.0, 0.0, -Inf, 0.0, 0.0, -Inf, -Inf, -Inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -Inf, 0.0, 0.0, 0.0, -Inf, 0.0, -Inf, -Inf, -Inf, -Inf, -Inf, 0.0, -Inf, 0.0, -Inf, -Inf, 0.0, -Inf, -Inf, 0.0, 0.0, 0.0, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, 0.0, -Inf, -Inf, -Inf, 0.0, -Inf], 2, 2, 2, 2, 2, 2)
    x2_table = reshape([0.0, -Inf, -Inf, -Inf, 0.0, 0.0, -Inf, 0.0, 0.0, -Inf, 0.0, -Inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -Inf, 0.0, 0.0, 0.0, -Inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -Inf, 0.0, 0.0, 0.0, -Inf, -Inf, 0.0, -Inf, -Inf, 0.0, -Inf, 0.0, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, 0.0, -Inf, -Inf, -Inf, 0.0, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf], 2, 2, 2, 2, 2, 2)

    for _ in 1:N
        f1 = VariableNode(length(variable_nodes), :f1)
        f1.support = [0,1]
        f1_factor = FactorNode([f1], [log(0.4), log(0.6)])
        push!(variable_nodes, f1)
        push!(factor_nodes, f1_factor)

        f2 = VariableNode(length(variable_nodes), :f2)
        f2.support = [0,1]
        f2_factor = FactorNode([f2], [log(0.4), log(0.6)])
        push!(variable_nodes, f2)
        push!(factor_nodes, f2_factor)

        d = VariableNode(length(variable_nodes), :d)
        d_factor = FactorNode([d], [log(1-0.001), log(0.001)])
        d.support = [0,1]
        push!(variable_nodes, d)
        push!(factor_nodes, d_factor)

        old_x1 = x1
        old_x2 = x2
        x1 = VariableNode(length(variable_nodes), :x1)
        x1_factor = FactorNode([old_x1, old_x2, f1, f2, d, x1], copy(x1_table))
        x1.support = [0,1]
        push!(variable_nodes, x1)
        push!(factor_nodes, x1_factor)

        x2 = VariableNode(length(variable_nodes), :x2)
        x2_factor = FactorNode([old_x1, old_x2, f1, f2, d, x2], copy(x2_table))
        x2.support = [0,1]
        push!(variable_nodes, x2)
        push!(factor_nodes, x2_factor)
    end

    marginal_variables = [x1.variable, x2.variable]

    for f in factor_nodes
        for v in f.neighbours
            push!(v.neighbours, f)
        end
    end

    return_variables = [net]
    return_factor = add_return_factor!(factor_nodes, return_variables)

    return variable_nodes, factor_nodes, marginal_variables, return_factor
end

function print_reference_solution(N=100)
    repeatf(n, f, x) = n > 1 ? f(repeatf(n-1, f, x)) : f(x)
    p_d = 0.001
    p_f = 0.6
    function T(P)
        t = (P[2,2] + P[2,1]) * (1 - p_d) + P[1,2]
        return [
            (P[1,1] + (P[2,2] + P[2,1]) * p_d) (1-p_f)*t;
            (p_f * t) 0
        ]
    end
    T0 = [0 0; 1 0]
    println("Reference: P=", repeatf(N, T, T0))
end
