function buildSparseId(T, Pspaces)
    TW = jordanmpotensortype(spacetype(Pspaces[1]), T)
    Os = map(Pspaces) do S
        O = TW(
            undef, SumSpace(fill(oneunit(S), 2)...) * S ← S * SumSpace(fill(oneunit(S), 2)...)
        )
        O[1, 1, 1, 1] = BraidingTensor{T}(eachspace(O)[1, 1, 1, 1])
        O[end, end, end, end] = BraidingTensor{T}(eachspace(O)[end, end, end, end])
        return O
    end
    return Os
end

"""
link_expectation(state, loc, F)

Calculate the expectation value of a link operator `F` at a given location `loc` in the state `state`.

# Arguments
`state`: The MPS state.
`loc::Int`: The location of the link operator.
`F::Function`: A function that takes a link representation and returns a scalar.
"""
function link_expectation(state, loc::Int, F::Function)
    S = zero(real(scalartype(state)))
    for (c, b) in blocks(state.C[loc])
        S += oftype(S, F(c) * dim(c) * tr(b * b'))
    end
    return S
end

link_expectation(state, locs, F::Function) = sum(link_expectation(state, loc, F) for loc in locs)

"""
link_expectation(state, ops)

Calculate the expectation value of a product of link operators acting on different
links of `state`.

# Arguments
- `state`: The MPS state.
- `ops`: A collection of pairs `loc::Int => F::Function` (e.g. `(n1 => F1, n2 => F2)`).

# Example
```julia
link_expectation(state, (n1 => F1, n2 => F2))
link_expectation(state, n1 => F1, n2 => F2)
```
"""
function link_expectation(state, ops)
    prs = sort(collect(ops); by = first)
    isempty(prs) && throw(ArgumentError("no link operators provided"))

    # A link operator at bond `loc` is diagonal in the bond charge sector `c`, acting
    # as multiplication by `F(c)`. To evaluate a product of such operators on distinct
    # bonds we transfer a bond environment through the left-canonical tensors between
    # consecutive links, applying each `F` at its bond, and finally close with the
    # center matrix. This reduces exactly to the single-link formula for one operator.
    first_loc = first(prs[1])
    E = MPSKit.l_LL(state, first_loc + 1) # identity on the first bond
    prev = first_loc
    for (loc, F) in prs
        for j in (prev + 1):loc
            E = E * TransferMatrix(state.AL[j])
        end
        for (t1, t2) in fusiontrees(E)
            E[t1, t2] .*= F(t1.uncoupled[1])
        end
        prev = loc
    end

    C = state.C[prev]
    return @tensor E[a; b] * C[b; c] * conj(C[a; c])
end

link_expectation(state, op::Pair, ops::Pair...) = link_expectation(state, (op, ops...))

function attenuateLinks(ψ::InfiniteMPS, reps, eta)
    rr = [x isa AbstractVector ? x : [x] for x in reps]
    ALs = deepcopy(ψ.AL)
    for i = eachindex(ALs)
        flag = false
        for (t1, t2) = fusiontrees(ALs[i])
            if t2.uncoupled[1] in rr[i]
                flag = true
            else
                ALs[i][t1, t2] .*= eta
            end
        end
        !flag && @warn "Target rep not found on link $i"
    end

    return InfiniteMPS(ALs)
end