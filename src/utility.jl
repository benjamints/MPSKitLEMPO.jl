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

function link_expectation(state, loc::Int, F::Function)
    S = zero(real(scalartype(state)))
    for (c, b) in entanglement_spectrum(state, loc)
        S += oftype(S, F(c) * dim(c) * sum(x^2 for x in b))
    end
    return S
end
