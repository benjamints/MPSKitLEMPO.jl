struct LEMPO_AC2 <: DerivativeOperator
    AC2::JordanMPO_AC2_Hamiltonian
    F::Union{Missing, Function}
    LE::TensorMap
    RE::TensorMap
end

function MPSKit.C_hamiltonian(
    site::Int, below::_HAM_MPS_TYPES, operator::Union{FiniteLEMPOHamiltonian, InfiniteLEMPOHamiltonian}, 
    above::_HAM_MPS_TYPES, envs)
    return MPSKit.MPO_C_Hamiltonian(leftenv(envs, site, below) * TransferMatrix(below.AL[site], operator[site], below.AL[site]), rightenv(envs, site, below))
end

function MPSKit.AC_hamiltonian(
        site::Int, below::_HAM_MPS_TYPES, operator::Union{FiniteLEMPOHamiltonian, InfiniteLEMPOHamiltonian},
        above::_HAM_MPS_TYPES, envs
    )
    return AC_hamiltonian(site, below, operator.mpo, above, envs)
end

function MPSKit.AC2_hamiltonian(
        site::Int, below::_HAM_MPS_TYPES, operator::Union{FiniteLEMPOHamiltonian, InfiniteLEMPOHamiltonian},
        above::_HAM_MPS_TYPES, envs
    )
    return LEMPO_AC2(AC2_hamiltonian(site, below, operator.mpo, above, envs), operator.link_fcts[site], leftenv(envs, site, below)[1], rightenv(envs, site + 1, below)[end])
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
