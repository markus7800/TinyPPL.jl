
using TinyPPL.Graph
using BenchmarkTools

function get_model()
    @ppl uninvoked Ladder begin
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
        @iterate(100, func, (1., 0.))
    end
end

function print_reference_solution()
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
    println("Reference: P=", repeatf(100, T, T0))
end
