module TestMPSKitModels
using ..TestSetup
using Test
using MPSKit
using MPSKitModels
using MPSKitLEMPO
using TensorKit
using SUNRepresentations
using LinearAlgebra

@testset "Fermions with SU(N) flavor symmetry" begin
    for N in 1:4, L = 2:3
        lat = FiniteChain(L)
        A = randn(ComplexF64, L, L)
        A = A + A'

        verts = vertices(lat)
        H = @mpoham sum(A[i, i] * staggered_c⁺cSUN(N, isodd(i)){i} for i in verts) + sum(A[i, j] * staggered_c⁺cSUN(N, isodd(i), isodd(j)){i,j} + A[j, i] * staggered_c⁺cSUN(N, isodd(j), isodd(i)){j,i} for (i, j) in [verts[i] => verts[j] for i in 1:length(lat) for j in (i+1):length(lat)])
        D, _ = eigen(convert(TensorMap, H))
        l1 = sort(real.(diag(convert(Array, D))))
        l2 = sort(energies(repeat(eigh_vals(A), N)))

        @test l1 ≈ l2
    end
end

@testset "Fermions without flavor symmetry" begin
    for N in 1:4, L = 2:3
        lat = FiniteChain(L)
        A = randn(ComplexF64, N, L, L)
        A = A + permutedims(conj(A), (1, 3, 2))

        verts = vertices(lat)
        H = @mpoham sum(A[k, i, i] * staggered_c⁺cN(k, N, isodd(i)){i} for k = 1:N, i in verts) + sum(A[k, i, j] * staggered_c⁺cN(k, N, isodd(i), isodd(j)){i,j} + A[k, j, i] * staggered_c⁺cN(k, N, isodd(j), isodd(i)){j,i} for k = 1:N, (i, j) in [verts[i] => verts[j] for i in 1:length(lat) for j in (i+1):length(lat)])
        D, _ = eigen(convert(TensorMap, H))
        l1 = sort(real.(diag(convert(Array, D))))
        l2 = sort(energies(vcat((eigh_vals(A[i, :, :]) for i = 1:N)...)))

        @test l1 ≈ l2
    end
end

@testset "DMRG2: Fermions with vs without flavor symmetry" begin
    for N = 1:3, L = 8, mlat = 0.5
        lat = FiniteChain(L)
        HN = @mpoham sum(mlat * (isodd(i) ? -1 : 1) * staggered_c⁺cN(k, N, isodd(i)){i} - 0.5im * (staggered_c⁺cN(k, N, isodd(i), isodd(j)){i,j} - staggered_c⁺cN(k, N, isodd(i), isodd(j))'{i,j}) for k = 1:N, (i, j) in nearest_neighbours(lat))
        HSUN = @mpoham sum(mlat * (isodd(i) ? -1 : 1) * staggered_c⁺cSUN(N, isodd(i)){i} - 0.5im * (staggered_c⁺cSUN(N, isodd(i), isodd(j)){i,j} - staggered_c⁺cSUN(N, isodd(i), isodd(j))'{i,j}) for (i, j) in nearest_neighbours(lat))

        ψN₀ = FiniteMPS(physicalspace(HN), Vect[(FermionParity⊠Irrep[U₁])]((f, k) => 2 for k = -1:1, f = 0:1))
        ψN, _ = find_groundstate(ψN₀, HN, DMRG2(; trscheme=truncerror(; atol=1.0e-8), verbosity=0))

        ψSUN₀ = FiniteMPS(physicalspace(HSUN), Vect[(FermionParity⊠Irrep[U₁]⊠SUNIrrep{N})]((f, k, [l == t ? 1 : 0 for l = 1:N-1]) => 2 for k = -1:1, f = 0:1, t = 0:N-1))
        ψSUN, a, b = find_groundstate(ψSUN₀, HSUN, DMRG2(; trscheme=truncerror(; atol=1.0e-8), verbosity=0))

        isapprox(expectation_value(ψN, HN), expectation_value(ψSUN, HSUN))

        @test expectation_value(ψN, HN) ≈ expectation_value(ψSUN, HSUN)
    end
end

end
