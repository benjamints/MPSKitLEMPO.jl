# Finite LEMPO Hamiltonians

Here we illustrate how to work with finite LEMPOs in `MPSKitLEMPO.jl`. Our motivating example will be $\text{SU}(2)$ lattice gauge theory with massless spin-$j$ Dirac fermions. After performing a Jordan-Wigner transformation to write the Hamiltonian in terms of bosonic operators, the Hamiltonian is
```math
H = \frac{g^2 a}{2}\sum_{n=1}^{N-1} L_n^A L_n^A - \frac{1}{2a}\sum_{n=1}^{N-1} S_n^{A} U_n^{AB} S_{n+1}^{B}.
```
Here, $S_n^A$ are spin-$j$ versions of Pauli matrices (indexed by the adjoint index $A=1,2,3$, with matrix indices suppressed). We will take $N$ to be even (otherwise, it is not possible to have $j$ be half-integer).

We first review how to construct the second term, the matter Hamiltonian, as an MPO in `MPSKit.jl`. The MPO representation of this Hamiltonian is given by
```math
H = \begin{pmatrix} 1 & 0 & 0 \end{pmatrix} \begin{pmatrix} I_1 & -\frac{1}{2a}S_1^{A_1} & \\ & & S_1^{A_0} \\ & & I_1 \end{pmatrix} \begin{pmatrix} I_2 & -\frac{1}{2a}S_2^{A_2} & \\ & & S_2^{A_1} \\ & & I_2 \end{pmatrix} \cdots \begin{pmatrix} I_n & -\frac{1}{2a}S_n^{A_n} & \\ & & S_n^{A_{n-1}} \\ & & I_n \end{pmatrix} \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}.
```
Note that the $U_n$ operators have been suppressed here; indeed, they can all be set to 1 by an appropriate gauge transformation. When we enter an MPO in `MPSKit.jl`, we do not need to explicitly include $U_n$; by using symmetric matrix product states, we guarantee local gauge invariance automatically.

The following function builds this MPO in `MPSKit.jl` (note that we work with the dimensionless operator $H/g$, and so the dimensionless parameter is $ga$):
```@example finite; output = false
using MPSKit, TensorKit

function su2_matter(N::Int; spin::Real = 1 / 2, ga::Real = 1.0)
    SL = ones(
        SU2Space(0 => 1) ⊗ SU2Space(spin => 1) ←
            SU2Space(spin => 1) ⊗ SU2Space(1 => 1)
    )
    SR = ones(
        SU2Space(1 => 1) ⊗ SU2Space(spin => 1) ←
            SU2Space(spin => 1) ⊗ SU2Space(0 => 1)
    )

    Elt = Union{Missing, typeof(SL), scalartype(SL)}
    A = Vector{Matrix{Elt}}(undef, N)

    for n in 1:N
        W = Matrix{Elt}(missing, 3, 3) # (f, SS, i)
        W[1, 1] = 1.0
        W[end, end] = 1.0
        W[3, 3] = 1.0

        W[2, end] = SR
        W[1, 2] = -1/(2ga) * SL

        A[n] = W
    end
    A[1] = A[1][1:1, :]
    A[N] = A[N][:, end:end]

    return FiniteMPOHamiltonian(A)
end
```

## Creating a `FiniteLEMPOHamiltonian`

To construct a `FiniteLEMPOHamiltonian`, we first construct a `FiniteMPOHamiltonian` (for instance, using the function above). We then build a vector of functions that take appropriate kinds of representations as arguments and return scalars.

```@example finite; output = false
using MPSKitLEMPO, MPSKit, TensorKit

function su2_full(N::Int; spin::Real = 1 / 2, ga::Real = 1.0)
    H_matter = su2_matter(N; spin, ga)

    F(r::SU2Irrep) = 0.5 * ga * (r.j * (r.j + 1) / 2)

    return FiniteLEMPOHamiltonian(H_matter, fill(F, N - 1))
end
```

## Using with DMRG

Finite LEMPO Hamiltonians can be used with standard DMRG algorithms from `MPSKit.jl`. As an example, it is straightforward to show that in the limit of large $ga$, our example Hamiltonian has a ground state energy of
```math
\frac{E_0}{g^2 a} =  \frac{1}{2} j(j+1) + \frac{1}{2}\begin{cases} (N-3) (j/2) (j/2+1)/2 & 2j = 0\pmod{4} \\ (N-4) (j^2 + 2j + 1/4)/8 + (j^2 + j - 3/4)/8 & 2j = 1\pmod{4} \\ (N-4) (j+1)^2/8 + (j^2 - 1)/8 & 2j = 2\pmod{4}  \\ (N-4) (j^2 + 2j + 1/4)/8 + (j^2 + 3j + 5/4)/8 & 2j = 3\pmod{4}  \end{cases}
```
This formula comes from minimizing the link energy. On the first and last link, we have to have spin-$j$; on the links in between, we can minimize the energy with an alternating pattern of spins (unless $j$ is an even integer, in which case we have spin-$j/2$ uniformly).

```@example finite
function gs_energy(N::Int; spin::Real = 1/2, ga::Real = 1.0)
    H = su2_full(N; spin, ga)
    physical_space = SU2Space(spin => 1)
    virtual_space = SU2Space([i => 10 for i=0:1/2:spin]...)
    ψ₀ = FiniteMPS(N, physical_space, virtual_space)

    ψ, _, _ = find_groundstate(ψ₀, H, DMRG2(; trscheme = truncerror(; atol = 1.0e-12)))

    return real(expectation_value(ψ, H))
end

results = []
gas = 4:4:20
spins = reverse(collect(1/2:1/2:3))
for spin in spins
    spin_results = []
    for ga in gas
        E = gs_energy(20; spin, ga)
        push!(spin_results, E/ga)
    end
    push!(results, spin_results)
end

using Plots, LaTeXStrings

plot(; xlabel=L"1/ga", ylabel=L"E_0/g^2a", legend=:topright, grid=true)
colors = [:blue, :red, :green, :purple, :orange, :brown]
for (i, spin) in enumerate(spins)
    plot!(1 ./ gas, results[i]; label="spin = $spin", color=colors[i], marker=:circle)
end
hline!([15/8, 5, 63/8, 23/2, 131/8, 45/2]; linestyle=:dash, color=:gray, label="")
```