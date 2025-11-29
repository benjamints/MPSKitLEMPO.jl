function MPSKit.effective_excitation_hamiltonian(
        H::InfiniteLEMPOHamiltonian, ϕ::QP,
        envs = environments(ϕ, H),
        energy = effective_excitation_renormalization_energy(H, ϕ, envs.leftenvs, envs.rightenvs)
    )
    ϕ′ = similar(ϕ)
    tforeach(1:length(ϕ); scheduler = Defaults.scheduler[]) do loc
        return ϕ′[loc] = _effective_excitation_local_apply(loc, ϕ, H, energy[loc], envs)
    end
    return ϕ′
end

function MPSKit.effective_excitation_renormalization_energy(H::InfiniteLEMPOHamiltonian, ϕ, lenvs, renvs)
    ψ_left = ϕ.left_gs
    ψ_right = ϕ.right_gs
    E = Vector{scalartype(ϕ)}(undef, length(ϕ))
    for i in eachindex(E)
        E[i] = contract_mpo_expval(
            ψ_left.AC[i], leftenv(lenvs, i, ψ_left) * LinkTransferMatrix(H.Fs[i - 1]), H[i], rightenv(lenvs, i, ψ_left)
        )
        if istopological(ϕ)
            E[i] += contract_mpo_expval(
                ψ_right.AC[i], leftenv(renvs, i, ψ_right) * LinkTransferMatrix(H.Fs[i - 1]), H[i], rightenv(renvs, i, ψ_right)
            )
            E[i] /= 2
        end
    end
    return E
end

function MPSKit._effective_excitation_local_apply(site, ϕ, H::InfiniteLEMPOHamiltonian, E::Number, envs)
    B = ϕ[site]
    GL = leftenv(envs.leftenvs, site, ϕ.left_gs) * LinkTransferMatrix(H.Fs[site - 1])
    GR = rightenv(envs.rightenvs, site, ϕ.right_gs)

    # renormalize first -> allocates destination
    B′ = scale(B, -E)

    # B in center
    @plansor B′[-1 -2; -3 -4] += GL[-1 5; 4] * B[4 2; -3 1] * H[site][5 -2; 2 3] * GR[1 3; -4]

    # B to the left
    if site > 1 || ϕ isa InfiniteQP
        AR = ϕ.right_gs.AR[site]
        GBL = envs.leftBenvs[site] * LinkTransferMatrix(H.Fs[site - 1])
        @plansor B′[-1 -2; -3 -4] += GBL[-1 4; -3 5] * AR[5 2; 1] * H[site][4 -2; 2 3] * GR[1 3; -4]
    end

    # B to the right
    if site < length(ϕ.left_gs) || ϕ isa InfiniteQP
        AL = ϕ.left_gs.AL[site]
        GBR = envs.rightBenvs[site]
        @plansor B′[-1 -2; -3 -4] += GL[-1 2; 1] * AL[1 3; 4] * H[site][2 -2; 3 5] * GBR[4 5; -3 -4]
    end

    return B′
end

function MPSKit.left_excitation_transfer_system(
        GBL, H::InfiniteLEMPOHamiltonian, exci;
        mom = exci.momentum, solver = Defaults.linearsolver
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
                        LinkTransferMatrix(H.Fs[div(i + 1, 2) - 1]) :
                        TransferMatrix(exci.right_gs.AR[div(i + 1, 2)], H[div(i + 1, 2)][:, 1, 1, :], exci.left_gs.AL[div(i + 1, 2)])
                        for i in 1:(2 * length(H))
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
        mom = exci.momentum,
        solver = Defaults.linearsolver
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
                        LinkTransferMatrix(H.Fs[div(i + 1, 2) - 1]) :
                        TransferMatrix(exci.left_gs.AL[div(i + 1, 2)], H[div(i + 1, 2)][:, 1, 1, :], exci.right_gs.AR[div(i + 1, 2)])
                        for i in 1:(2 * length(H))
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
