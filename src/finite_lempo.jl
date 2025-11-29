struct FiniteLEMPOHamiltonian{O} <: AbstractMPO{O}
    mpo::FiniteMPOHamiltonian{O}
    Fs::Vector{Union{Missing, Function}}

    function FiniteLEMPOHamiltonian(mpo::FiniteMPOHamiltonian{O}, Fs) where {O}
        if length(Fs) != length(mpo) - 1
            throw(ArgumentError("Lengths do not match"))
        end
        return new{O}(mpo, Fs)
    end
end

function FiniteLEMPOHamiltonian(T::Type, Pspaces, Fs)
    if length(Fs) != length(Pspaces) - 1
        throw(ArgumentError("Lengths do not match"))
    end
    Ws = buildSparseId(T,Pspaces)
    Ws[1] = Ws[1][1:1, :, :, :]
    Ws[end] = Ws[end][:, :, :, end:end]
    return FiniteLEMPOHamiltonian(FiniteMPOHamiltonian(Ws), Fs)
end
FiniteLEMPOHamiltonian(Pspaces, Fs) = FiniteLEMPOHamiltonian(Float64, Pspaces, Fs)

Base.parent(x::FiniteLEMPOHamiltonian) = x.mpo.W

function MPSKit.expectation_value(
        ψ::FiniteMPS, H::FiniteLEMPOHamiltonian,
        envs::AbstractMPSEnvironments = environments(ψ, H)
    )
    return dot(ψ, H.mpo, ψ, envs) / dot(ψ, ψ)
end
