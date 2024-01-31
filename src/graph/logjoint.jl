
function make_logjoint(pgm::PGM)
    function logjoint(X::AbstractVector{<:Real})
        return pgm.logpdf(X, pgm.observations)
    end
    return logjoint
end

function make_unconstrained_logjoint(pgm::PGM)
    function logjoint(X::AbstractVector{<:Real})
        return pgm.unconstrained_logpdf(X, pgm.observations)
    end
    return logjoint
end