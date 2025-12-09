module TestSetup
using TensorKit, MPSKit, MPSKitLEMPO

export finite_heisenberg_gauss, finite_heisenberg_link, infinite_heisenberg_link, infinite_heisenberg
export infinite_z2_gauge, infinite_z2_spin, energies

function energies(l)
    L = length(l)
    sums = zeros(2^L)
    for (i, n) in enumerate(Iterators.product(ntuple(_ -> 0:1, L)...))
        sums[i] = sum(l[i] * n[i] for i in 1:L)
    end
    return sums
end

function infinite_heisenberg(N::Int; spin::Real=1 / 2)

    SL = -spin * (spin + 1) *
         ones(
             SU2Space(0 => 1) ⊗ SU2Space(spin => 1) ←
             SU2Space(spin => 1) ⊗ SU2Space(1 => 1)
         )
    SR = ones(
        SU2Space(1 => 1) ⊗ SU2Space(spin => 1) ←
        SU2Space(spin => 1) ⊗ SU2Space(0 => 1)
    )

    Elt = Union{Missing,typeof(SL),scalartype(SL)}
    A = Vector{Matrix{Elt}}(undef, N)

    for n in 1:N
        W = Matrix{Elt}(missing, 3, 3) # (f, SS, i)
        W[1, 1] = 1.0
        W[end, end] = 1.0
        W[3, 3] = 1.0

        W[2, end] = SR
        W[1, 2] = SL

        A[n] = W
    end

    return InfiniteMPOHamiltonian(A)
end

function finite_heisenberg_link(N::Int; spin::Real=1 / 2, g::Real=1.0)
    F(r) = 0.5 * g * r.j * (r.j + 1)

    SL = -spin * (spin + 1) *
         ones(
             SU2Space(0 => 1) ⊗ SU2Space(spin => 1) ←
             SU2Space(spin => 1) ⊗ SU2Space(1 => 1)
         )
    SR = ones(
        SU2Space(1 => 1) ⊗ SU2Space(spin => 1) ←
        SU2Space(spin => 1) ⊗ SU2Space(0 => 1)
    )

    Elt = Union{Missing,typeof(SL),scalartype(SL)}
    A = Vector{Matrix{Elt}}(undef, N)

    for n in 1:N
        W = Matrix{Elt}(missing, 3, 3) # (f, SS, i)
        W[1, 1] = 1.0
        W[end, end] = 1.0
        W[3, 3] = 1.0

        W[2, end] = SR
        W[1, 2] = SL

        A[n] = W
    end
    A[1] = A[1][1:1, :]
    A[N] = A[N][:, end:end]

    HH = FiniteMPOHamiltonian(A)

    return FiniteLEMPOHamiltonian(HH, fill(F, N - 1))
end

function infinite_heisenberg_link(N::Int; spin::Real=1 / 2, g::Real=1.0)
    F(r) = 0.5 * g * r.j * (r.j + 1)

    SL = -spin * (spin + 1) *
         ones(
             SU2Space(0 => 1) ⊗ SU2Space(spin => 1) ←
             SU2Space(spin => 1) ⊗ SU2Space(1 => 1)
         )
    SR = ones(
        SU2Space(1 => 1) ⊗ SU2Space(spin => 1) ←
        SU2Space(spin => 1) ⊗ SU2Space(0 => 1)
    )

    Elt = Union{Missing,typeof(SL),scalartype(SL)}
    A = Vector{Matrix{Elt}}(undef, N)

    for n in 1:N
        W = Matrix{Elt}(missing, 3, 3) # (f, SS, i)
        W[1, 1] = 1.0
        W[end, end] = 1.0
        W[3, 3] = 1.0

        W[2, end] = SR
        W[1, 2] = SL

        A[n] = W
    end

    HH = InfiniteMPOHamiltonian(A)

    return InfiniteLEMPOHamiltonian(HH, fill(F, N))
end


function finite_heisenberg_gauss(N::Int; spin::Real=1 / 2, g::Real=1.0, J::Real=1.0)
    SL = -spin * (spin + 1) *
         ones(
             SU2Space(0 => 1) ⊗ SU2Space(spin => 1) ←
             SU2Space(spin => 1) ⊗ SU2Space(1 => 1)
         )
    SR = ones(
        SU2Space(1 => 1) ⊗ SU2Space(spin => 1) ←
        SU2Space(spin => 1) ⊗ SU2Space(0 => 1)
    )

    Elt = Union{Missing,typeof(SL),scalartype(SL)}
    A = Vector{Matrix{Elt}}(undef, N)

    for n in 1:N
        W = Matrix{Elt}(missing, 4, 4) # (f, SS, QQ, i)
        W[1, 1] = 1.0
        W[end, end] = 1.0
        W[3, 3] = 1.0

        W[2, end] = SR
        W[1, 2] = SL

        W[3, end] = g * (N + 1 - n) * SR
        W[1, 3] = SL
        W[1, end] = 0.5 * g * (N + 1 - n) * 0.75

        A[n] = W
    end
    A[1] = A[1][1:1, :]
    A[N] = A[N][:, end:end]

    return FiniteMPOHamiltonian(A)
end

function infinite_z2_gauge(N::Int; g::Real=1.0, J::Real=1.0, h::Real=1.0)
    SL = ones(ℤ₂Space(0 => 1) ⊗ ℤ₂Space(0 => 1, 1 => 1) ← ℤ₂Space(0 => 1, 1 => 1) ⊗ ℤ₂Space(1 => 1))
    SR = ones(ℤ₂Space(1 => 1) ⊗ ℤ₂Space(0 => 1, 1 => 1) ← ℤ₂Space(0 => 1, 1 => 1) ⊗ ℤ₂Space(0 => 1))
    SZ = ones(ℤ₂Space(0 => 1) ⊗ ℤ₂Space(0 => 1, 1 => 1) ← ℤ₂Space(0 => 1, 1 => 1) ⊗ ℤ₂Space(0 => 1))
    block(SZ, Irrep[ℤ₂](0)) .= -1.0

    Elt = Union{Missing,typeof(SL),scalartype(SL)}
    A = Vector{Matrix{Elt}}(undef, N)

    for n in 1:N
        W = Matrix{Elt}(missing, 3, 3) # (f, SS, i)
        W[1, 1] = 1.0
        W[end, end] = 1.0

        W[2, end] = SR
        W[1, 2] = J * SL
        W[1, end] = h * SZ

        A[n] = W
    end

    HH = InfiniteMPOHamiltonian(A)

    F(r) = r.n == 0 ? -g : g
    return InfiniteLEMPOHamiltonian(HH, fill(F, N))
end

function infinite_z2_spin(N::Int; g::Real=1.0, J::Real=1.0, h::Real=1.0)
    X = TensorMap([0.0 1.0; 1.0 0.0], ℂ^1 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^1)
    Z = TensorMap([1.0 0.0; 0.0 -1.0], ℂ^1 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^1)

    Elt = Union{Missing,typeof(X),scalartype(X)}
    A = Vector{Matrix{Elt}}(undef, N)

    for n in 1:N
        W = Matrix{Elt}(missing, 3, 3) # (f, SS, i)
        W[1, 1] = 1.0
        W[end, end] = 1.0


        W[2, end] = -h * Z # Why is there a sign here?
        W[1, 2] = Z
        W[1, end] = g * Z + J * X

        A[n] = W
    end

    return InfiniteMPOHamiltonian(A)
end

end
