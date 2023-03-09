module Distributions
    import Distributions: Distribution, logpdf, params, mean

    import Distributions: Bernoulli, Binomial, Categorical, DiscreteUniform, Geometric, Poisson
    import Distributions: Beta, Cauchy, Exponential, Gamma, InverseGamma, Laplace, LogNormal, Normal, TDist, Uniform
    import Distributions: Dirac

    include("../utils/utils.jl")
    include("logpdf_grad.jl")
    include("proposal.jl")
    include("random_walk.jl")


    export Distribution, logpdf, params, mean

    export Bernoulli, Binomial, Categorical, DiscreteUniform, Geometric, Poisson
    export Beta, Cauchy, Exponential, Gamma, InverseGamma, Laplace, LogNormal, Normal, TDist, Uniform
    export Dirac
end