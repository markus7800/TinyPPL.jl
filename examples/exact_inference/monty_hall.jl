function get_model()
    @ppl MontyHall begin
        let C ~ Categorical([1/3, 1/3, 1/3]), # contestant, can be arbitrary
            P ~ Categorical([1/3, 1/3, 1/3]), # prize
            H ~ if C == 1. && P == 1. # host
                Categorical([0, 1/2, 1/2]) # can be arbitrary (except 0)
            elseif C == 2. && P == 2.
                Categorical([1/2, 0, 1/2]) # can be arbitrary (except 0)
            elseif C == 3. && P == 3.
                Categorical([1/2, 1/2, 0]) # can be arbitrary (except 0)
            elseif C == 1. && P == 2.
                Categorical([0, 0, 1])
            elseif C == 1. && P == 3.
                Categorical([0, 1, 0])
            elseif C == 2. && P == 1.
                Categorical([0, 0, 1])
            elseif C == 2. && P == 3.
                Categorical([1, 0, 0])
            elseif C == 3. && P == 1.
                Categorical([0, 1, 0])
            elseif C == 3. && P == 2.
                Categorical([1, 0, 0])
            end,
            S ~ Dirac(P != C) # should switch?

            S
        end
    end
end

function print_reference_solution()
    println("Reference: ", "P(0)=", 1/3, " P(1)=", 2/3)
end