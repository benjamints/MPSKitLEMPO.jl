module TestQP
using ..TestSetup
using Test
using MPSKit
using MPSKitLEMPO
using TensorKit

@testset "QP: Z2 gauge theory, link vs spin" begin

    for N in [1], ss in [5], g in [0.3, 0.5, 1.5], J in [0.2, 0.6, 2.5], ks in 2 * pi * [[0, 0.4, 0.7]]
        HS = infinite_z2_spin(N; g = g, J = J)
        HL = infinite_z2_gauge(N; g = g, J = J)

        virtual_space = ℤ₂Space(0 => ss, 1 => ss)
        ψ₀ = InfiniteMPS(physicalspace(HL), fill(virtual_space, N))
        ψ, _ = find_groundstate(ψ₀, HL, VUMPS(; verbosity = 0))
        EL, _ = excitations(HL, QuasiparticleAnsatz(), 0, ψ; num = 1)

        virtual_space = ℂ^(2 * ss)
        ψ₀ = InfiniteMPS(physicalspace(HS), fill(virtual_space, N))
        ψ, _ = find_groundstate(ψ₀, HS, VUMPS(; verbosity = 0))
        ES, _ = excitations(HS, QuasiparticleAnsatz(), 0, ψ; num = 1)

        @test ES ≈ EL
    end
end

end
