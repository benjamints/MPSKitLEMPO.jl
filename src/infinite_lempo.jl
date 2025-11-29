struct InfiniteLEMPOHamiltonian{O} <: AbstractMPO{O}
    mpo::InfiniteMPOHamiltonian{O}
    Fs::PeriodicVector{Union{Missing, Function}}

    function InfiniteLEMPOHamiltonian(mpo::InfiniteMPOHamiltonian{O}, Fs) where {O}
        if length(Fs) != length(mpo)
            throw(ArgumentError("Lengths do not match"))
        end
        return new{O}(mpo, Fs)
    end
end

function InfiniteLEMPOHamiltonian(T, Pspaces, Fs)
    if length(Fs) != length(Pspaces)
        throw(ArgumentError("Lengths do not match"))
    end
    return InfiniteLEMPOHamiltonian(InfiniteMPOHamiltonian(buildSparseId(T, Pspaces)), Fs)
end
InfiniteLEMPOHamiltonian(Pspaces, Fs) = InfiniteLEMPOHamiltonian(Float64, Pspaces, Fs)

Base.parent(x::InfiniteLEMPOHamiltonian) = x.mpo.W
Base.repeat(H::InfiniteLEMPOHamiltonian{T}, n::Int) where {T} = InfiniteLEMPOHamiltonian(repeat(H.mpo, n), repeat(H.Fs, n))

function MPSKit.expectation_value(
        ψ::InfiniteMPS, H::InfiniteLEMPOHamiltonian,
        envs::AbstractMPSEnvironments = environments(ψ, H)
    )
    return sum(1:length(ψ)) do i
        return contract_mpo_expval(
            ψ.AC[i], envs.GLs[i] * LinkTransferMatrix(H.Fs[i - 1]), H[i][:, 1, 1, end], envs.GRs[i][end]
        )
    end
end
