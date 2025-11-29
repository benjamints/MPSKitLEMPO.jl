function MPSKit.environments(below::FiniteMPS, O::FiniteLEMPOHamiltonian, above = nothing)
    env = environments(below, O.mpo, above)
    return FiniteEnvironments(env.above, O, env.ldependencies, env.rdependencies, env.GLs, env.GRs)
end

function MPSKit.rightenv(ca::FiniteEnvironments{A, B, C, D}, ind, state) where {A, B <: FiniteLEMPOHamiltonian, C, D}
    a = findfirst(i -> !(state.AR[i] === ca.rdependencies[i]), length(state):-1:(ind + 1))
    a = isnothing(a) ? nothing : length(state) - a + 1

    if !isnothing(a)
        #we need to recalculate
        for j in a:-1:(ind + 1)
            above = isnothing(ca.above) ? state.AR[j] : ca.above.AR[j]
            ca.GRs[j] = LinkTransferMatrix(ca.operator.Fs[j - 1]) * (TransferMatrix(above, ca.operator[j], state.AR[j]) *
                ca.GRs[j + 1])
            ca.rdependencies[j] = state.AR[j]
        end
    end

    return ca.GRs[ind + 1]
end

function MPSKit.leftenv(ca::FiniteEnvironments{A, B, C, D}, ind, state) where {A, B <: FiniteLEMPOHamiltonian, C, D}
    a = findfirst(i -> !(state.AL[i] === ca.ldependencies[i]), 1:(ind - 1))

    if !isnothing(a)
        #we need to recalculate
        for j in a:(ind - 1)
            above = isnothing(ca.above) ? state.AL[j] : ca.above.AL[j]
            ca.GLs[j + 1] = (
                ca.GLs[j] *
                    TransferMatrix(above, ca.operator[j], state.AL[j])
            ) * LinkTransferMatrix(ca.operator.Fs[j])
            ca.ldependencies[j] = state.AL[j]
        end
    end

    return ca.GLs[ind]
end
