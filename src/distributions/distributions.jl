module Distributions
    import Distributions: Distribution, logpdf, params, mean, mode

    import Distributions: Bernoulli, Binomial, Categorical, DiscreteUniform, Geometric, Poisson
    import Distributions: Beta, Cauchy, Exponential, Gamma, InverseGamma, Laplace, LogNormal, Normal, TDist, Uniform
    import Distributions: Dirac

    include("logpdf_grad.jl")
    include("proposal.jl")
    include("random_walk.jl")
    include("transformed.jl")
    include("variational.jl")
    include("elbo.jl")


    export Distribution, logpdf, params, mean, mode

    export Bernoulli, Binomial, Categorical, DiscreteUniform, Geometric, Poisson
    export Beta, Cauchy, Exponential, Gamma, InverseGamma, Laplace, LogNormal, Normal, TDist, Uniform
    export Dirac
end