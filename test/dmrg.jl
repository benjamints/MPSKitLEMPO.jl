module TestDMRG
    using ..TestSetup
    using Test
    using MPSKit
    using MPSKitLEMPO
    using TensorKit

    @testset "DMRG1 Heisenberg gauge theory" begin
        for N in [10], g in [0.5, 1.0, 3.0], spin in [1 / 2, 1, 3 / 2]
            HL = finite_heisenberg_link(N; g = g, spin = spin)
            HG = finite_heisenberg_gauss(N; g = g, spin = spin)

            virtual_space = SU2Space(l => 2 for l in 0:0.5:3)
            ψ₀ = FiniteMPS(N, SU2Space(spin => 1), virtual_space)

            ψL, envs, delta = find_groundstate(ψ₀, HL, DMRG(; verbosity = 0))
            ψG, envs, delta = find_groundstate(ψ₀, HG, DMRG(; verbosity = 0))

            @test abs(dot(ψL, ψG)) ≈ 1
        end
    end

    @testset "DMRG2 Heisenberg gauge theory" begin
        for N in [10], g in [0.5, 1.0, 3.0], spin in [1 / 2, 1, 3 / 2]
            HL = finite_heisenberg_link(N; g = g, spin = spin)
            HG = finite_heisenberg_gauss(N; g = g, spin = spin)

            virtual_space = SU2Space(l => 2 for l in 0:0.5:3)
            ψ₀ = FiniteMPS(N, SU2Space(spin => 1), virtual_space)

            ψL, envs, delta = find_groundstate(ψ₀, HL, DMRG2(; trscheme = truncerror(; atol = 1.0e-8), verbosity = 0))
            ψG, envs, delta = find_groundstate(ψ₀, HG, DMRG2(; trscheme = truncerror(; atol = 1.0e-8), verbosity = 0))

            @test abs(dot(ψL, ψG)) ≈ 1
        end
    end
end
