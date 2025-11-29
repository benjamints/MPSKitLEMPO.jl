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

end
