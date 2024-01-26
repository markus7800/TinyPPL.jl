function get_model()
    @pgm MurderMystery begin
        function mystery()
            let aliceDunnit ~ Bernoulli(0.3),
                withGun ~ (aliceDunnit == 1 ? Bernoulli(0.03) : Bernoulli(0.8))
                (aliceDunnit, withGun)
            end
        end
        function gunFoundAtScene(gunFound)
            let t = mystery(),
                aliceDunnit = t[1],
                withGun = t[2]

                Dirac(withGun) ↦ 1.
                aliceDunnit
            end
        end
        let posterior = gunFoundAtScene(true)
            posterior
        end
    end
end

function print_reference_solution()
    println("Reference: ", "P(0)=", 1-9/569, " P(1)=", 9/569)
end