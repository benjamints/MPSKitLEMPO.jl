"""
`left_link(envs, H)`

Acts on the left environment of `envs` with the link transfer matrix of the LEMPO `H`, returning new environments.

# Arguments
- `envs`: The infinite environments.
- `H`: The infinite LEMPO Hamiltonian.
"""
function left_link(envs::InfiniteEnvironments, H::InfiniteLEMPOHamiltonian)
    GLs = similar(envs.GLs)
     for i in eachindex(envs.GLs)
        GLs[i] = envs.GLs[i] * LinkTransferMatrix(H.Fs[i - 1])
    end
    return InfiniteEnvironments(GLs, envs.GRs)
end

# Conventions: 
# GL = ... * (link) * (MPO)
# GR = (link) * (MPO) * ...
# So when computing expecation values, derivatives etc. GL should be multiplied by link operator in addition to A
function MPSKit.environments(
        below::InfiniteMPS, operator::InfiniteLEMPOHamiltonian,
        above::InfiniteMPS = below; kwargs...
    )
    GLs, GRs = initialize_environments(below, operator.mpo, above)
    envs = InfiniteEnvironments(GLs, GRs)
    return recalculate!(envs, below, operator, above; kwargs...)
end

function MPSKit.recalculate!(
        envs::InfiniteEnvironments, below::InfiniteMPS,
        operator::InfiniteLEMPOHamiltonian,
        above::InfiniteMPS = below;
        kwargs...
    )
    if !issamespace(envs, below, operator.mpo, above)
        # TODO: in-place initialization?
        GLs, GRs = initialize_environments(below, operator.mpo, above)
        copy!(envs.GLs, GLs)
        copy!(envs.GRs, GRs)
    end

    alg = environment_alg(below, operator.mpo, above; kwargs...)

    @sync begin
        @spawn compute_leftenvs!(envs, below, operator, above, alg)
        @spawn compute_rightenvs!(envs, below, operator, above, alg)
    end
    # normalize!(envs, below, operator, above)

    return envs
end

function MPSKit.compute_leftenvs!(
        envs::InfiniteEnvironments, below::InfiniteMPS,
        operator::InfiniteLEMPOHamiltonian, above::InfiniteMPS, alg
    )
    L = check_length(below, above, operator.mpo)
    GLs = envs.GLs
    vsize = length(first(GLs))

    @assert above === below "not implemented"

    ρ_left = l_LL(above)
    ρ_right = r_LL(above)

    # the start element
    # TODO: check if this is necessary
    # leftutil = similar(above.AL[1], space(GL[1], 2)[1])
    # fill_data!(leftutil, one)
    # @plansor GL[1][1][-1 -2; -3] = ρ_left[-1; -3] * leftutil[-2]

    (L > 1) && left_cyclethrough!(1, GLs, below, operator, above)

    for i in 2:vsize
        prev = copy(GLs[1][i])
        zerovector!(GLs[1][i])
        left_cyclethrough!(i, GLs, below, operator, above)

        if isidentitylevel(operator.mpo, i) # identity matrices; do the hacky renormalization
            T = regularize(TransferMatrix(above.AL, below.AL), ρ_left, ρ_right)
            GLs[1][i], convhist = linsolve(flip(T), GLs[1][i], prev, alg, 1, -1)
            convhist.converged == 0 &&
                @warn "GL$i failed to converge: normres = $(convhist.normres)"

            (L > 1) && left_cyclethrough!(i, GLs, below, operator, above)

            # go through the unitcell, again subtracting fixpoints
            for site in 1:L
                @plansor GLs[site][i][-1 -2; -3] -= GLs[site][i][1 -2; 2] *
                    r_LL(above, site - 1)[2; 1] * l_LL(above, site)[-1; -3]
            end

        else
            if !isemptylevel(operator.mpo, i)
                diag = map(h -> h[i, 1, 1, i], operator.mpo[:])
                T = TransferMatrix(above.AL, diag, below.AL)
                GLs[1][i], convhist = linsolve(flip(T), GLs[1][i], prev, alg, 1, -1)
                convhist.converged == 0 &&
                    @warn "GL$i failed to converge: normres = $(convhist.normres)"
            end
            (L > 1) && left_cyclethrough!(i, GLs, below, operator, above)
        end
    end

    # for i in eachindex(GLs)
    #     GLs[i] = GLs[i] * LinkTransferMatrix(operator.Fs[i - 1])
    # end

    return GLs
end

function MPSKit.left_cyclethrough!(
        index::Int, GL, below::InfiniteMPS, H::InfiniteLEMPOHamiltonian,
        above::InfiniteMPS = below
    )
    # TODO: efficient transfer matrix slicing for large unitcells
    vsize = length(first(GL))
    leftinds = 1:index
    for site in eachindex(GL)
        if index == vsize
            B = GL[site] * LinkTransferMatrix(H.Fs[site])
            GL[site + 1][index] = B * TransferMatrix(
                above.AL[site], H[site][leftinds, 1, 1, index], below.AL[site]
            )
        else
            GL[site + 1][index] = GL[site][leftinds] * TransferMatrix(
                above.AL[site], H[site][leftinds, 1, 1, index], below.AL[site]
            )
        end
    end
    return GL
end

function MPSKit.compute_rightenvs!(
        envs::InfiniteEnvironments, below::InfiniteMPS,
        operator::InfiniteLEMPOHamiltonian, above::InfiniteMPS, alg
    )
    L = check_length(above, operator.mpo, below)
    GRs = envs.GRs
    vsize = length(last(GRs))

    @assert above === below "not implemented"

    ρ_left = l_RR(above)
    ρ_right = r_RR(above)

    # the start element
    # TODO: check if this is necessary
    # rightutil = similar(state.AL[1], space(GR[end], 2)[end])
    # fill_data!(rightutil, one)
    # @plansor GR[end][end][-1 -2; -3] = r_RR(state)[-1; -3] * rightutil[-2]

    (L > 1) && right_cyclethrough!(vsize, GRs, below, operator, above) # populate other sites

    for i in (vsize - 1):-1:1
        prev = copy(GRs[end][i])
        zerovector!(GRs[end][i])
        right_cyclethrough!(i, GRs, below, operator, above)

        if isidentitylevel(operator.mpo, i) # identity matrices; do the hacky renormalization
            # subtract fixpoints
            T = regularize(TransferMatrix(above.AR, below.AR), ρ_left, ρ_right)
            GRs[end][i], convhist = linsolve(T, GRs[end][i], prev, alg, 1, -1)
            convhist.converged == 0 &&
                @warn "GR$i failed to converge: normres = $(convhist.normres)"

            L > 1 && right_cyclethrough!(i, GRs, below, operator, above)

            # go through the unitcell, again subtracting fixpoints
            for site in 1:L
                @plansor GRs[site][i][-1 -2; -3] -= GRs[site][i][1 -2; 2] *
                    l_RR(above, site + 1)[2; 1] * r_RR(above, site)[-1; -3]
            end
        else
            if !isemptylevel(operator.mpo, i)
                diag = map(b -> b[i, 1, 1, i], operator.mpo[:])
                T = TransferMatrix(above.AR, diag, below.AR)
                GRs[end][i], convhist = linsolve(T, GRs[end][i], prev, alg, 1, -1)
                convhist.converged == 0 &&
                    @warn "GR$i failed to converge: normres = $(convhist.normres)"
            end

            (L > 1) && right_cyclethrough!(i, GRs, below, operator, above)
        end
    end

    return GRs
end

function MPSKit.right_cyclethrough!(
        index::Int, GR, below::InfiniteMPS, operator::InfiniteLEMPOHamiltonian,
        above::InfiniteMPS = below
    )
    # TODO: efficient transfer matrix slicing for large unitcells
    for site in reverse(eachindex(GR))
        rightinds = index:length(GR[site])
        GR[site - 1][index] = TransferMatrix(
            above.AR[site], operator[site][index, 1, 1, rightinds], below.AR[site]
        ) * GR[site][rightinds]

        if index == 1
            GR[site - 1] = LinkTransferMatrix(operator.Fs[site]) * GR[site - 1]
        end
    end
    return GR
end
