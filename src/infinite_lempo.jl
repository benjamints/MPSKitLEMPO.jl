"""
`InfiniteLEMPOHamiltonian(mpo::InfiniteMPOHamiltonian, Fs::PeriodicVector{Union{Missing, Function}})`

Constructs an infinite LEMPO, represented as an MPO together with a vector of link functions.

# Arguments:
- `mpo`: The infinite MPO Hamiltonian.
- `Fs`: A periodic vector of link functions, where each function takes a link representation and returns a scalar (or is `missing`, if the function is zero).
"""
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

"""
`InfiniteLEMPOHamiltonian(T, Pspaces, Fs::PeriodicVector{Union{Missing, Function}})`

Constructs an infinite LEMPO Hamiltonian that only acts nontrivially on links.

# Arguments
- `T`: The type of the scalars in the Hamiltonian.
- `Pspaces`: A periodic vector of physical spaces.
- `Fs`: A periodic vector of link functions, where each function takes a link representation and returns a scalar (or is `missing`, if the function is zero).
"""
function InfiniteLEMPOHamiltonian(T, Pspaces, Fs)
    if length(Fs) != length(Pspaces)
        throw(ArgumentError("Lengths do not match"))
    end
    return InfiniteLEMPOHamiltonian(InfiniteMPOHamiltonian(buildSparseId(T, Pspaces)), Fs)
end
InfiniteLEMPOHamiltonian(Pspaces, Fs) = InfiniteLEMPOHamiltonian(Float64, Pspaces, Fs)

Base.isfinite(x::InfiniteLEMPOHamiltonian) = false
Base.parent(x::InfiniteLEMPOHamiltonian) = x.mpo.W
Base.repeat(H::InfiniteLEMPOHamiltonian{T}, n::Int) where {T} = InfiniteLEMPOHamiltonian(repeat(H.mpo, n), repeat(H.Fs, n))

"""
`MPSKit.expectation_value(ψ::InfiniteMPS, H::InfiniteLEMPOHamiltonian, envs::AbstractMPSEnvironments = environments(ψ, H))`

Calculates the expectation value of an infinite LEMPO `H` in the infinite MPS state `ψ`, optionally using the provided environments `envs`.

# Arguments
- `ψ`: The infinite MPS state.
- `H`: The infinite LEMPO Hamiltonian.
- `envs`: The environments for the MPS state (default: `environments(ψ, H)`).
"""
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
