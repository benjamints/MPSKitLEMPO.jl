module TestLink
using ..TestSetup
using Test
using MPSKit
using MPSKitLEMPO
using TensorKit

@testset "Link only expectation" begin
    F(r) = 0.5 * r.j * (r.j + 1)
    V = SU2Space(s => 2 for s in 0:0.5:3)
    for N in [1, 3], spin in [1 / 2, 3 / 2]
        P = SU2Space(spin => 1)
        ψ = InfiniteMPS(fill(P, N), fill(V, N))

        for k in 1:N
            x1 = link_expectation(ψ, k, F)
            x2 = expectation_value(ψ, InfiniteLEMPOHamiltonian(physicalspace(ψ), [k == l ? F : missing for l in 1:N]))
            @test x1 ≈ x2
        end
    end
end

@testset "Link expectation as part of Heisenberg" begin
    F(r) = 0.5 * r.j * (r.j + 1)
    V = SU2Space(s => 2 for s in 0:0.5:3)
    for N in [1, 3], spin in [1 / 2, 3 / 2]
        P = SU2Space(spin => 1)
        ψ = InfiniteMPS(fill(P, N), fill(V, N))

        H = infinite_heisenberg_link(N; spin = spin)
        HH = infinite_heisenberg(N; spin = spin)

        x1 = expectation_value(ψ, H) - expectation_value(ψ, HH)
        x2 = sum(link_expectation(ψ, k, F) for k in 1:N)
        @test x1 ≈ x2
    end
end

@testset "Product of link operators" begin
    # F is the SU(2) Casimir, so a link operator on link `n` measures j_n(j_n+1), where
    # j_n is the representation on link `n`. By Gauss' law this equals the total block
    # spin (∑_{i≤n} Sᵢ)² of the physical chain, so a product of link operators must agree
    # with the expectation value of the product of block-Casimir operators.
    F(r) = r.j * (r.j + 1)
    N = 6
    virtual_space = SU2Space(l => 2 for l in 0:0.5:3)
    for spin in [1 / 2, 1]
        P = SU2Space(spin => 1)
        lattice = fill(P, N)
        ψ₀ = FiniteMPS(N, P, virtual_space)
        H = finite_heisenberg_link(N; spin = spin)
        ψ, = find_groundstate(ψ₀, H, DMRG(; verbosity = 0))

        # a single-link product reduces to the plain single-link expectation value
        for n in 1:(N-1)
            @test link_expectation(ψ, (n => F,)) ≈ link_expectation(ψ, n, F)
        end

        # products of link operators match the Gauss-law (block-spin) computation
        for (n1, n2) in [(2, 4), (1, 5), (3, 3)]
            x_link = link_expectation(ψ, (n1 => F, n2 => F))
            C1 = block_casimir(lattice, n1; spin = spin)
            C2 = block_casimir(lattice, n2; spin = spin)
            x_gauss = expectation_value(ψ, C1 * C2)
            @test x_link ≈ x_gauss
        end
    end
end

end
