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