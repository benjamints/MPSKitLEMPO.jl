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

function Base.:*(H::FiniteLEMPOHamiltonian, mps::FiniteMPS)
    N = check_length(H, mps)
    @assert N > 2 "MPS should have at least three sites, to be implemented otherwise"
    A = convert.(BlockTensorMap, [mps.AC[1]; mps.AR[2:end]])
    A′ = similar(
        A,
        tensormaptype(
            spacetype(mps), numout(eltype(mps)), numin(eltype(mps)),
            promote_type(scalartype(H), scalartype(mps))
        )
    )
    # left to middle
    U = ones(scalartype(H), left_virtualspace(H, 1))
    OL = link_mpo_tensor(
        scalartype(H), right_virtualspace(mps, 1), right_virtualspace(H.mpo, 1), H.link_fcts[1]
    )
    @plansor a[-1 -2; -3 -4] := A[1][-1 2; 4] * H[1][1 -2; 2 5] * OL[4 5; -3 -4] * conj(U[1])
    Q, R = qr_compact!(a)
    A′[1] = TensorMap(Q)

    for i in 2:(N ÷ 2)
        OL = link_mpo_tensor(
            scalartype(H), right_virtualspace(mps, i), right_virtualspace(H.mpo, i), H.link_fcts[i]
        )
        @plansor a[-1 -2; -3 -4] := R[-1; 1 2] * A[i][1 3; 4] * H[i][2 -2; 3 5] * OL[4 5; -3 -4]
        Q, R = qr_compact!(a)
        A′[i] = TensorMap(Q)
    end

    # right to middle
    U = ones(scalartype(H), right_virtualspace(H, N))
    @plansor a[-1 -2; -3 -4] := A[end][-1 2; -3] * H[end][-2 -4; 2 1] * U[1] # no OL on the right end
    L, Q = lq_compact!(a)
    A′[end] = transpose(TensorMap(Q), ((1, 3), (2,)))

    for i in (N - 1):-1:(N ÷ 2 + 2)
        OL = link_mpo_tensor(
            scalartype(H), right_virtualspace(mps, i), right_virtualspace(H.mpo, i), H.link_fcts[i]
        )
        @plansor a[-1 -2; -3 -4] := A[i][-1 3; 1] * H[i][-2 -4; 3 2] * OL[1 2; 4 5] * L[4 5; -3]
        L, Q = lq_compact!(a)
        A′[i] = transpose(TensorMap(Q), ((1, 3), (2,)))
    end

    # connect pieces
    OL = link_mpo_tensor(
        scalartype(H), right_virtualspace(mps, N ÷ 2 + 1), right_virtualspace(H.mpo, N ÷ 2 + 1),
        H.link_fcts[N ÷ 2 + 1]
    )
    @plansor a[-1 -2; -3] := R[-1; 1 2] * A[N ÷ 2 + 1][1 3; 4] * H[N ÷ 2 + 1][2 -2; 3 5] * OL[4 5; 6 7] * L[6 7; -3]
    A′[N ÷ 2 + 1] = TensorMap(a)

    without_end = FiniteMPS(A′)
    if ismissing(H.link_fcts[N])
        return without_end
    else
        last_irrep = keys(right_virtualspace(mps, N).dims)[1]
        end_part = H.link_fcts[N](last_irrep) * mps
        return without_end + end_part
    end
end

function link_mpo_tensor(T, mps_vspace, mpo_vspace, link_fct::Union{Missing,Function})
    OL = zeros(T, mps_vspace ⊗ mpo_vspace ← mps_vspace ⊗ mpo_vspace)
    for i=1:length(mpo_vspace)
        OL[1,i,1,i] = id(T, mps_vspace ⊗ mpo_vspace[i])
    end
    if !ismissing(link_fct)
        OL[1,1,1,end] = id(T, mps_vspace ⊗ mpo_vspace[1])
        for (t1, t2) in fusiontrees(OL[1,1,1,end])
            OL[1,1,1,end][t1, t2] .*= link_fct(t1.uncoupled[1])
        end
    end
    return OL
end