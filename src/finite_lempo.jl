"""
`FiniteLEMPOHamiltonian(mpo::FiniteMPOHamiltonian, link_fcts::Vector{Union{Missing, Function}})`

Constructs a finite LEMPO, represented as an MPO together with a vector of link functions.

# Arguments:
- `mpo`: The finite MPO Hamiltonian.
- `link_fcts`: A vector of link functions, where each function takes a link representation and returns a scalar (or is `missing`, if the function is zero).
"""
struct FiniteLEMPOHamiltonian{O} <: AbstractMPO{O}
    mpo::FiniteMPOHamiltonian{O}
    link_fcts::Vector{Union{Missing, Function}}

    function FiniteLEMPOHamiltonian(mpo::FiniteMPOHamiltonian{O}, link_fcts) where {O}
        if length(link_fcts) < length(mpo)
            link_fcts = vcat(link_fcts, fill(missing, length(mpo) - length(link_fcts)))
        end
        if length(link_fcts) != length(mpo)
            throw(ArgumentError("Length of link_fcts ($(length(link_fcts))) must not exceed length of mpo ($(length(mpo)))."))
        end
        return new{O}(mpo, link_fcts)
    end
end

"""
`FiniteLEMPOHamiltonian(T, Pspaces, link_fcts::Vector{Union{Missing, Function}})`

Constructs a finite LEMPO Hamiltonian that only acts nontrivially on links.

# Arguments
- `T`: The type of the scalars in the Hamiltonian.
- `Pspaces`: A vector of physical spaces.
- `link_fcts`: A vector of link functions, where each function takes a link representation and returns a scalar (or is `missing`, if the function is zero).
"""
function FiniteLEMPOHamiltonian(T::Type, Pspaces, link_fcts)
    Ws = buildSparseId(T,Pspaces)
    Ws[1] = Ws[1][1:1, :, :, :]
    Ws[end] = Ws[end][:, :, :, end:end]
    return FiniteLEMPOHamiltonian(FiniteMPOHamiltonian(Ws), link_fcts)
end
FiniteLEMPOHamiltonian(Pspaces, link_fcts) = FiniteLEMPOHamiltonian(Float64, Pspaces, link_fcts)

Base.parent(x::FiniteLEMPOHamiltonian) = x.mpo.W
Base.isfinite(x::FiniteLEMPOHamiltonian) = true

function MPSKit.expectation_value(
        ψ::FiniteMPS, H::FiniteLEMPOHamiltonian,
        envs::AbstractMPSEnvironments = environments(ψ, H)
    )
    dynamic_part = dot(ψ, H.mpo, ψ, envs) / dot(ψ, ψ)

    last_irrep = keys(right_virtualspace(ψ, length(ψ)).dims)[1]
    end_part = ismissing(H.link_fcts[end]) ? 0. : H.link_fcts[end](last_irrep)
    
    return dynamic_part + end_part
end
