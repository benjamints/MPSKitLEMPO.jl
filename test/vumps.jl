module TestVUMPS
using ..TestSetup
using Test
using MPSKit
using MPSKitLEMPO
using TensorKit

@testset "VUMPS: Z2 gauge theory, link vs spin" begin

    for N in [1], k in [3, 5, 7], g in [0.3, 0.5, 1.5], J in [0.2, 0.6, 2.5]
        HS = infinite_z2_spin(N; g = g, J = J)
        HL = infinite_z2_gauge(N; g = g, J = J)

        virtual_space = ℤ₂Space(0 => k, 1 => k)
        ψ₀ = InfiniteMPS(physicalspace(HL), fill(virtual_space, N))
        ψ, _ = find_groundstate(ψ₀, HL, VUMPS(; verbosity = 0))
        e1 = expectation_value(ψ, HL)

        virtual_space = ℂ^(2 * k)
        ψ₀ = InfiniteMPS(physicalspace(HS), fill(virtual_space, N))
        ψ, _ = find_groundstate(ψ₀, HS, VUMPS(; verbosity = 0))
        e2 = expectation_value(ψ, HS)

        @test e1 ≈ e2
    end
end

end
