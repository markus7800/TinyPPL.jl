
using TinyPPL.Distributions
using TinyPPL.Graph
import Random


model = @pgm simple begin
    let X ~ Normal(0., 1.)
        Normal(X, 1.) ↦ 1.
        X
    end
end

