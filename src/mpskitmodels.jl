Base.isodd(l::LatticePoint) = isodd(sum(l.coordinates))
Base.iseven(l::LatticePoint) = iseven(sum(l.coordinates))

function casimir(irrep::SUNIrrep{N}) where N
    L = dynkin_label(irrep)
    return sum(
        sum(
            ((min(i, j) * (N - max(i, j))) / N) *
            L[i] * (L[j] + 2)
            for j in 1:N-1
        )
        for i in 1:N-1
    )/2
end


function staggered_cSUN(N, oddsite)
    if oddsite
        P = Vect[FermionParity⊠U1Irrep⊠SUNIrrep{N}]((isodd(k - N), k - N, [m == k ? 1 : 0 for m in 1:(N-1)]) => 1 for k in 0:N)
    else
        P = Vect[FermionParity⊠U1Irrep⊠SUNIrrep{N}]((isodd(k), k, [m == k ? 1 : 0 for m in 1:(N-1)]) => 1 for k in 0:N)
    end
    V = Vect[FermionParity⊠U1Irrep⊠SUNIrrep{N}]((1, 1, [m == 1 ? 1 : 0 for m in 1:(N-1)]) => 1)

    cR = zeros(ComplexF64, V ⊗ P ← P)
    for (s, b) in blocks(cR)
        b .= sqrt(s[2].charge + (oddsite ? N : 0))
    end
    return cR
end

function staggered_c⁺cSUN(N, oddsite1, oddsite2)
    cL = staggered_cSUN(N, oddsite1)
    cR = staggered_cSUN(N, oddsite2)

    return @tensor T[-1 -2; -3 -4] := conj(cL[1 -3; -1]) * cR[1 -2; -4]
end

function staggered_c⁺cSUN(N, oddsite)
    cL = staggered_cSUN(N, oddsite)
    cR = staggered_cSUN(N, oddsite)

    return @tensor T[-1; -2] := conj(cL[1 2; -1]) * cR[1 2; -2]
end

function staggered_cN(k, N, oddsite)
    if oddsite
        P = Vect[FermionParity⊠U1Irrep]((1, -1) => 1, (0, 0) => 1)
    else
        P = Vect[FermionParity⊠U1Irrep]((0, 0) => 1, (1, 1) => 1)
    end
    V = Vect[FermionParity⊠U1Irrep]((1, 1) => 1)
    c = ones(ComplexF64, V ⊗ P ← P)

    T = foldr(⊗, [k == l ? c : id(P) for l = 1:N])
    if k > 1
        T = permute(T, (tuple((i == 1 ? k : (i < k + 1 ? i - 1 : i) for i = 1:N+1)...), tuple((i for i = N+2:2*N+1)...)))
    end
    return (id(V) ⊗ isomorphism(fuse(P^N), P^N)) * T * isomorphism(P^N, fuse(P^N))
end

function staggered_c⁺cN(k, N, oddsite1, oddsite2)
    cL = staggered_cN(k, N, oddsite1)
    cR = staggered_cN(k, N, oddsite2)

    return @tensor T[-1 -2; -3 -4] := conj(cL[1 -3; -1]) * cR[1 -2; -4]
end

function staggered_c⁺cN(k, N, oddsite)
    cL = staggered_cN(k, N, oddsite)
    cR = staggered_cN(k, N, oddsite)

    return @tensor T[-1; -2] := conj(cL[1 2; -1]) * cR[1 2; -2]
end