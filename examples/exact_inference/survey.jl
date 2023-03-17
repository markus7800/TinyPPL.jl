
function get_model()
    @ppl Survey begin
        let A ~ Categorical([0.3, 0.5, 0.2]), # yound adult old
            S ~ Categorical([0.6, 0.4]), # M, F
            E = if (A == 1 && S == 1) # high, uni
                Categorical([0.75, 0.25])
            elseif (A == 2 && S == 1)
                Categorical([0.72, 0.28])
            elseif (A == 3 && S == 1)
                Categorical([0.88, 0.12])
            elseif (A == 1 && S == 2)
                Categorical([0.64, 0.36])
            elseif (A == 2 && S == 2)
                Categorical([0.7, 0.3])
            elseif (A == 3 && S == 2)
                Categorical([0.9, 0.1])
            end ↦ 1,
            R ~ if (E == 1) # small, big
                Categorical([0.25, 0.75])
            else
                Categorical([0.2, 0.8])
            end,
            O ~ if (E == 1) # emp, self
                Categorical([0.96, 0.04])
            else
                Categorical([0.92, 0.08])
            end,
            T ~ if (O == 1 && R == 1) # car, train, other
                Categorical([0.48, 0.42, 0.1])
            elseif (O == 2 && R == 1)
                Categorical([0.56, 0.36, 0.08])
            elseif (O == 1 && R == 2)
                Categorical([0.58, 0.24, 0.18])
            elseif (O == 2 && R == 2)
                Categorical([0.7, 0.21, 0.09])
            end
            
            (S, T)
        end
    end
end

function print_reference_solution()
    println("Reference: ", "P=", [0.3426644 0.1736599 0.09623271; 0.2167356 0.1098401 0.06086729])
end
