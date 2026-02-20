function GrassmannMPS.fg(
        state::InfiniteMPS, operator::Union{O, LazySum{O}},
        envs::AbstractMPSEnvironments = environments(state, operator)
    ) where {O <: InfiniteLEMPOHamiltonian}
    recalculate!(envs, state, operator, state)
    f = expectation_value(state, operator, envs)
    isapprox(imag(f), 0; atol = eps(abs(f))^(3 / 4)) || @warn "MPO might not be Hermitian: $f"

    A = Core.Compiler.return_type(Grassmann.project, Tuple{eltype(state), eltype(state)})
    gs = Vector{A}(undef, length(state))
    tmap!(gs, 1:length(state); scheduler = MPSKit.Defaults.scheduler[]) do i
        AC′ = AC_hamiltonian(i, state, operator, state, envs) * state.AC[i]
        g = Grassmann.project(AC′, state.AL[i])
        return rmul(g, state.C[i]')
    end
    return real(f), gs
end