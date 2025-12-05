struct LEMPO_AC2 <: DerivativeOperator
    AC2::JordanMPO_AC2_Hamiltonian
    F::Union{Missing, Function}
    LE::TensorMap
    RE::TensorMap
end

struct EnvironmentsContainer{V <: GenericMPSTensor}
    GL::V
    GR::V
end

function EnvironmentsContainer(GL, GR)
    return EnvironmentsContainer{typeof(V)}(GL, GR)
end

MPSKit.leftenv(envs::EnvironmentsContainer, site::Int, state) = envs.GL
MPSKit.rightenv(envs::EnvironmentsContainer, site::Int, state) = envs.GR

function MPSKit.AC_hamiltonian(
        site::Int, below::_HAM_MPS_TYPES, operator::FiniteLEMPOHamiltonian,
        above::_HAM_MPS_TYPES, envs
    )
    return AC_hamiltonian(site, below, operator.mpo, above, envs)
end

function MPSKit.AC2_hamiltonian(
        site::Int, below::_HAM_MPS_TYPES, operator::FiniteLEMPOHamiltonian,
        above::_HAM_MPS_TYPES, envs
    )
    return LEMPO_AC2(AC2_hamiltonian(site, below, operator.mpo, above, envs), operator.Fs[site], leftenv(envs, site, below)[1], rightenv(envs, site + 1, below)[end])
end

function MPSKit.AC_hamiltonian(
        site::Int, below::_HAM_MPS_TYPES, operator::InfiniteLEMPOHamiltonian,
        above::_HAM_MPS_TYPES, envs
    )
    envC = EnvironmentsContainer(leftenv(envs, site, below) * LinkTransferMatrix(operator.Fs[site - 1]), rightenv(envs, site, below))
    return AC_hamiltonian(site, below, operator.mpo, above, envC)
end

function MPSKit.AC2_hamiltonian(
        site::Int, below::_HAM_MPS_TYPES, operator::InfiniteLEMPOHamiltonian,
        above::_HAM_MPS_TYPES, envs
    )
    envC = EnvironmentsContainer(leftenv(envs, site, below) * LinkTransferMatrix(operator.Fs[site - 1]), rightenv(envs, site + 1, below))
    if ismissing(operator.Fs[site])
        return AC2_hamiltonian(site, below, operator.mpo, above, envC)
    else
        return LEMPO_AC2(AC2_hamiltonian(site, below, operator.mpo, above, envC), operator.Fs[site], envC.GL[1], envC.GR[end])
    end
end

function (P::LEMPO_AC2)(v::MPOTensor)
    if ismissing(P.F)
        return P.AC2(v)
    else
        x = copy(v)
        for (s, b) in blocks(x)
            b .*= P.F(s)
        end
        @tensor toret[-1 -2; -3 -4] := P.LE[-1 2; 6] * x[6 -2; 1 -4] *
            P.RE[1 2; -3]

        return P.AC2(v) + toret
    end
end
