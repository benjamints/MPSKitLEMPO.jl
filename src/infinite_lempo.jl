"""
`InfiniteLEMPOHamiltonian(mpo::InfiniteMPOHamiltonian, link_fcts::PeriodicVector{Union{Missing, Function}})`

Constructs an infinite LEMPO, represented as an MPO together with a vector of link functions.

# Arguments:
- `mpo`: The infinite MPO Hamiltonian.
- `link_fcts`: A periodic vector of link functions, where each function takes a link representation and returns a scalar (or is `missing`, if the function is zero).
"""
struct InfiniteLEMPOHamiltonian{O} <: AbstractMPO{O}
    mpo::InfiniteMPOHamiltonian{O}
    link_fcts::PeriodicVector{Union{Missing,Function}}

    function InfiniteLEMPOHamiltonian(mpo::InfiniteMPOHamiltonian{O}, link_fcts) where {O}
        if length(link_fcts) != length(mpo)
            throw(ArgumentError("Lengths do not match"))
        end
        return new{O}(mpo, link_fcts)
    end
end

"""
`InfiniteLEMPOHamiltonian(T, Pspaces, link_fcts::PeriodicVector{Union{Missing, Function}})`

Constructs an infinite LEMPO Hamiltonian that only acts nontrivially on links.

# Arguments
- `T`: The type of the scalars in the Hamiltonian.
- `Pspaces`: A periodic vector of physical spaces.
- `link_fcts`: A periodic vector of link functions, where each function takes a link representation and returns a scalar (or is `missing`, if the function is zero).
"""
function InfiniteLEMPOHamiltonian(T, Pspaces, link_fcts)
    if length(link_fcts) != length(Pspaces)
        throw(ArgumentError("Lengths do not match"))
    end
    return InfiniteLEMPOHamiltonian(InfiniteMPOHamiltonian(buildSparseId(T, Pspaces)), link_fcts)
end
InfiniteLEMPOHamiltonian(Pspaces, link_fcts) = InfiniteLEMPOHamiltonian(Float64, Pspaces, link_fcts)

Base.isfinite(x::InfiniteLEMPOHamiltonian) = false
Base.parent(x::InfiniteLEMPOHamiltonian) = x.mpo.W
Base.repeat(H::InfiniteLEMPOHamiltonian{T}, n::Int) where {T} = InfiniteLEMPOHamiltonian(repeat(H.mpo, n), repeat(H.link_fcts, n))

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
    envs::AbstractMPSEnvironments=environments(ψ, H)
)
    return sum(1:length(ψ)) do i
        R = (LinkTransferMatrix(H.link_fcts[i-1]) * (TransferMatrix(ψ.AC[i], H[i], ψ.AC[i]) * envs.GRs[i]))[1]
        L =  envs.GLs[i][1]
        @tensor res = L[1 2; 3] * R[3 2; 1]
        return res
    end
end
# function MPSKit.expectation_value(
#     ψ::InfiniteMPS, H::InfiniteLEMPOHamiltonian,
#     envs::AbstractMPSEnvironments=environments(ψ, H)
# )
#     return sum(1:length(ψ)) do i
#         return contract_mpo_expval(
#             ψ.AC[i], envs.GLs[i][1], H[i][1, 1, 1, :], LinkTransferMatrix(H.link_fcts[i]) * envs.GRs[i]
#         )
#     end
# end
