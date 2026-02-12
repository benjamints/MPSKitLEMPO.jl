function MPSKit.effective_excitation_hamiltonian(
    H::InfiniteLEMPOHamiltonian, ϕ::QP,
    envs=environments(ϕ, H),
    energy=effective_excitation_renormalization_energy(H, ϕ, envs.leftenvs, envs.rightenvs)
)
    ϕ′ = similar(ϕ)
    tforeach(1:length(ϕ); scheduler=Defaults.scheduler[]) do loc
        return ϕ′[loc] = _effective_excitation_local_apply(loc, ϕ, H, energy[loc], envs)
    end
    return ϕ′
end

function MPSKit._effective_excitation_local_apply(site, ϕ, H::InfiniteLEMPOHamiltonian, E::Number, envs)
    return _effective_excitation_local_apply(site, ϕ, H.mpo, E, envs)
end

function MPSKit.left_excitation_transfer_system(
    GBL, H::InfiniteLEMPOHamiltonian, exci;
    mom=exci.momentum, solver=Defaults.linearsolver
)
    len = length(H)
    found = zerovector(GBL)
    odim = length(GBL)

    if istrivial(exci)
        ρ_left = l_RL(exci.right_gs)
        ρ_right = r_RL(exci.right_gs)
    end

    for i in 1:odim
        # this operation can in principle be even further optimized for larger unit cells
        # as we only require the terms that end at level i.
        # this would require to check the finite state machine, and discard non-connected
        # terms.
        if i == odim
            T = ProductTransferMatrix(
                [
                isodd(i) ?
                LinkTransferMatrix(H.link_fcts[div(i + 1, 2)-1]) :
                TransferMatrix(exci.right_gs.AR[div(i + 1, 2)], H[div(i + 1, 2)][:, 1, 1, :], exci.left_gs.AL[div(i + 1, 2)])
                for i in 2:(2*length(H))+1
            ]
            )
        else
            H_partial = map(h -> getindex(h, 1:i, 1, 1, 1:i), parent(H))
            T = TransferMatrix(exci.right_gs.AR, H_partial, exci.left_gs.AL)
        end
        start = scale!(last(found[1:i] * T), cis(-mom * len))
        if istrivial(exci) && isidentitylevel(H.mpo, i)
            regularize!(start, ρ_right, ρ_left)
        end

        found[i] = add!(start, GBL[i])

        if !isemptylevel(H.mpo, i)
            if isidentitylevel(H.mpo, i)
                T = TransferMatrix(exci.right_gs.AR, exci.left_gs.AL)
                if istrivial(exci)
                    T = regularize(T, ρ_left, ρ_right)
                end
            else
                T = TransferMatrix(
                    exci.right_gs.AR, map(h -> h[i, 1, 1, i], parent(H)), exci.left_gs.AL
                )
            end

            found[i], convhist = linsolve(
                flip(T), found[i], found[i], solver, 1, -cis(-mom * len)
            )
            convhist.converged == 0 &&
                @warn "GBL$i failed to converge: normres = $(convhist.normres)"
        end
    end
    return found
end

function MPSKit.right_excitation_transfer_system(
    GBR, H::InfiniteLEMPOHamiltonian, exci;
    mom=exci.momentum,
    solver=Defaults.linearsolver
)
    len = length(H)
    found = zerovector(GBR)
    odim = length(GBR)

    if istrivial(exci)
        ρ_left = l_LR(exci.right_gs)
        ρ_right = r_LR(exci.right_gs)
    end

    for i in odim:-1:1
        # this operation can in principle be even further optimized for larger unit cells
        # as we only require the terms that end at level i.
        # this would require to check the finite state machine, and discard non-connected
        # terms.
        if i == 1
            T = ProductTransferMatrix(
                [
                isodd(i) ?
                LinkTransferMatrix(H.link_fcts[div(i + 1, 2)-1]) :
                TransferMatrix(exci.left_gs.AL[div(i + 1, 2)], H[div(i + 1, 2)][:, 1, 1, :], exci.right_gs.AR[div(i + 1, 2)])
                for i in 1:(2*length(H))
            ]
            )
        else
            H_partial = map(h -> h[i:end, 1, 1, i:end], parent(H))
            T = TransferMatrix(exci.left_gs.AL, H_partial, exci.right_gs.AR)
        end
        start = scale!(first(T * found[i:odim]), cis(mom * len))
        if istrivial(exci) && isidentitylevel(H.mpo, i)
            regularize!(start, ρ_left, ρ_right)
        end

        found[i] = add!(start, GBR[i])

        if !isemptylevel(H.mpo, i)
            if isidentitylevel(H.mpo, i)
                tm = TransferMatrix(exci.left_gs.AL, exci.right_gs.AR)
                if istrivial(exci)
                    tm = regularize(tm, ρ_left, ρ_right)
                end
            else
                tm = TransferMatrix(
                    exci.left_gs.AL, map(h -> h[i, 1, 1, i], parent(H)), exci.right_gs.AR
                )
            end

            found[i], convhist = linsolve(
                tm, found[i], found[i], solver, 1, -cis(mom * len)
            )
            convhist.converged < 1 &&
                @warn "GBR$i failed to converge: normres = $(convhist.normres)"
        end
    end
    return found
end
