function MPSKit.environments(exci::InfiniteQP, H::InfiniteLEMPOHamiltonian, lenvs, renvs; kwargs...)
    ids = findall(Base.Fix1(isidentitylevel, H.mpo), 2:(size(H[1], 1) - 1)) .+ 1
    solver = environment_alg(exci, H.mpo, exci; kwargs...)

    AL = exci.left_gs.AL
    AR = exci.right_gs.AR

    lBs = PeriodicVector([allocate_GBL(exci, H.mpo, exci, i) for i in 1:length(exci)])
    rBs = PeriodicVector([allocate_GBR(exci, H.mpo, exci, i) for i in 1:length(exci)])

    zerovector!(lBs[1])
    for pos in 1:length(exci)
        lBs[pos + 1] = (lBs[pos] * LinkTransferMatrix(H.Fs[pos - 1])) * TransferMatrix(AR[pos], H[pos], AL[pos]) /
            cis(exci.momentum)
        lBs[pos + 1] += (leftenv(lenvs, pos, exci.left_gs) * LinkTransferMatrix(H.Fs[pos - 1])) *
            TransferMatrix(exci[pos], H[pos], AL[pos]) / cis(exci.momentum)

        if istrivial(exci) && !isempty(ids) # regularization of trivial excitations
            ρ_left = l_RL(exci.left_gs, pos + 1)
            ρ_right = r_RL(exci.left_gs, pos)
            for i in ids
                regularize!(lBs[pos + 1][i], ρ_right, ρ_left)
            end
        end
    end

    zerovector!(rBs[end])
    for pos in length(exci):-1:1
        rBs[pos - 1] = TransferMatrix(AL[pos], H[pos], AR[pos]) *
            rBs[pos] * cis(exci.momentum)
        rBs[pos - 1] += TransferMatrix(exci[pos], H[pos], AR[pos]) *
            rightenv(renvs, pos, exci.right_gs) * cis(exci.momentum)

        rBs[pos - 1] = LinkTransferMatrix(H.Fs[pos - 1]) * rBs[pos - 1]

        if istrivial(exci) && !isempty(ids)
            ρ_left = l_LR(exci.left_gs, pos)
            ρ_right = r_LR(exci.left_gs, pos - 1)
            for i in ids
                regularize!(rBs[pos - 1][i], ρ_left, ρ_right)
            end
        end
    end

    @sync begin
        Threads.@spawn $lBs[1] = left_excitation_transfer_system(
            $lBs[1], $H, $exci; solver = $solver
        )
        Threads.@spawn $rBs[end] = right_excitation_transfer_system(
            $rBs[end], $H, $exci; solver = $solver
        )
    end

    lB_cur = lBs[1]
    for i in 1:(length(exci) - 1)
        lB_cur = (lB_cur * LinkTransferMatrix(H.Fs[i - 1])) * TransferMatrix(AR[i], H[i], AL[i]) / cis(exci.momentum)

        if istrivial(exci) && !isempty(ids)
            ρ_left = l_RL(exci.left_gs, i + 1)
            ρ_right = r_RL(exci.left_gs, i)
            for k in ids
                regularize!(lB_cur[k], ρ_right, ρ_left)
            end
        end

        lBs[i + 1] += lB_cur
    end

    rB_cur = rBs[end]
    for i in length(exci):-1:2
        rB_cur = LinkTransferMatrix(H.Fs[i - 1]) * (TransferMatrix(AL[i], H[i], AR[i]) * rB_cur) * cis(exci.momentum)

        if istrivial(exci) && !isempty(ids)
            ρ_left = l_LR(exci.left_gs, i)
            ρ_right = r_LR(exci.left_gs, i - 1)
            for k in ids
                regularize!(rB_cur[k], ρ_left, ρ_right)
            end
        end

        rBs[i - 1] += rB_cur
    end

    return InfiniteQPEnvironments(lBs, rBs, lenvs, renvs)
end
