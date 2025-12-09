module TestQP
using ..TestSetup
using Test
using MPSKit
using MPSKitModels
using MPSKitLEMPO
using TensorKit

@testset "QP: ZN gauge theory, link vs spin" begin
    for N in 2:4, L in 2:4
        g = -3 .+ PeriodicArray(rand(L))
        J = PeriodicArray(rand(L))
        h = PeriodicArray(rand(L))
        lat = InfiniteChain(L)

        H1 = @mpoham sum(J[i] * (ℤₙV⁺V(N){i,j} + ℤₙV⁺V(N)'{i,j}) + h[i] * (ℤₙU(N){i} + ℤₙU(N)'{i}) for (i, j) in nearest_neighbours(lat))
        HL = InfiniteLEMPOHamiltonian(H1, [r -> g[i] * ℤₙf(r) for i in 1:L])

        HS = @mpoham sum(g[i] * (ℂₙU(N){i} + ℂₙU(N)'{i}) + J[i] * (ℂₙV(N){i} + ℂₙV(N)'{i}) + h[i+1] * (ℂₙU(N)'{i} * ℂₙU(N){j} + ℂₙU(N){i} * ℂₙU(N)'{j}) for (i, j) in nearest_neighbours(lat))

        virtual_space = ZNSpace{N}(k => 2 for k in 1:N)
        ψ₀ = InfiniteMPS(physicalspace(HL), fill(virtual_space, length(HL)))
        ψ, _ = find_groundstate(ψ₀, HL, VUMPS(; verbosity=0, finalize=(iter, ψ, H, envs) -> changebonds(ψ, H, VUMPSSvdCut(; trscheme=truncerror(atol=1e-8)), envs), tol=1e-6))
        EL, _ = excitations(HL, QuasiparticleAnsatz(), 0, ψ; num=1)

        virtual_space = ℂ^(N * 2)
        ψ₀ = InfiniteMPS(physicalspace(HS), fill(virtual_space, length(HS)))
        ψ, _ = find_groundstate(ψ₀, HS, VUMPS(; verbosity=0, finalize=(iter, ψ, H, envs) -> changebonds(ψ, H, VUMPSSvdCut(; trscheme=truncerror(atol=1e-8)), envs), tol=1e-6))
        ES, _ = excitations(HS, QuasiparticleAnsatz(), 0, ψ; num=1)

        @test ES[1] ≈ EL[1]
    end
end

end
