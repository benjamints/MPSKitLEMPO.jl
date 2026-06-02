function MPSKit.environments(
        below::WindowMPS, O::InfiniteLEMPOHamiltonian, above = nothing;
        lenvs = MPSKit.environments(below.left_gs, O),
        renvs = MPSKit.environments(below.right_gs, O)
    )
    leftstart = copy(lenvs.GLs[1])
    rightstart = copy(renvs.GRs[end])

    return MPSKit.environments(below, O, above, leftstart, rightstart)
end