# Infinite LEMPO Hamiltonians

Here we illustrate how to work with infinite LEMPOs in `MPSKitLEMPO.jl`. As in the [finite case](../guide/finite_lempo.md), our motivating example will be $\text{SU}(2)$ lattice gauge theory with massless spin-$j$ Dirac fermions. The Hamiltonian is
```math
H = \frac{g^2 a}{2}\sum_{n} L_n^A L_n^A - \frac{1}{2a}\sum_{n} S_n^{A} U_n^{AB} S_{n+1}^{B}.
```
Here, $S_n^A$ are spin-$j$ versions of Pauli matrices (indexed by the adjoint index $A=1,2,3$, with matrix indices suppressed).

We first review how to construct the second term, the matter Hamiltonian, as an infinite MPO in `MPSKit.jl`. The setup is similar to the [finite case](../guide/finite_lempo.md), but here we use the manifest translation-invariance of the Hamiltonian explicitly.
```@example infinite; output = false
using MPSKit, TensorKit

function su2_matter(; spin::Real = 1 / 2, ga::Real = 1.0)
    SL = ones(
        SU2Space(0 => 1) ⊗ SU2Space(spin => 1) ←
            SU2Space(spin => 1) ⊗ SU2Space(1 => 1)
    )
    SR = ones(
        SU2Space(1 => 1) ⊗ SU2Space(spin => 1) ←
            SU2Space(spin => 1) ⊗ SU2Space(0 => 1)
    )

    Elt = Union{Missing, typeof(SL), scalartype(SL)}
    A = Vector{Matrix{Elt}}(undef, 2)

    for n in 1:2
        W = Matrix{Elt}(missing, 3, 3) # (f, SS, i)
        W[1, 1] = 1.0
        W[end, end] = 1.0
        W[3, 3] = 1.0

        W[2, end] = SR
        W[1, 2] = -1/(2ga) * SL

        A[n] = W
    end

    return InfiniteMPOHamiltonian(A)
end
```

## Creating an `InfiniteLEMPOHamiltonian`

To construct an `InfiniteLEMPOHamiltonian`, we first construct an `InfiniteMPOHamiltonian` (for instance, using the function above). We then build a vector of functions that take appropriate kinds of representations as arguments and return scalars.

```@example infinite; output = false
using MPSKitLEMPO, MPSKit, TensorKit

function su2_full(; spin::Real = 1 / 2, ga::Real = 1.0)
    H_matter = su2_matter(; spin, ga)

    F(r::SU2Irrep) = 0.5 * ga * (r.j * (r.j + 1) / 2)

    return InfiniteLEMPOHamiltonian(H_matter, fill(F, 2))
end
```

## Using with VUMPS

Infinite LEMPO Hamiltonians can be used with the VUMPS algorithm from `MPSKit.jl`. To run VUMPS, we need an initial infinite MPS. This can be subtle: it is important for the infinite MPS to be *injective*, which essentially means that it describes one state and not multiple states. For instance, in the spin-$1/2$ case, if we allow all representations on all links then we will end up describing a state that has integer spins on even links and another state that has half-integer spins on even links, all in the same infinite MPS. To avoid this, we will be careful about which representations we place on the links in the initial infinite MPS.

It is straightforward to show that in the limit of large $ga$, our example Hamiltonian has a ground state energy density (energy per two-site unit cell) of
```math
\frac{\varepsilon_0}{g^2 a} = \frac{1}{2}\begin{cases} (j/2)(j/2 + 1) & 2j = 0\pmod{4} \\ (j+1)^2/4 & 2j = 2\pmod{4} \\ (j^2 + 2j + 1/4)/8 & \text{otherwise} \end{cases}
```
This can be derived from the large-$N$ limit of the ground state energy formula given in the [finite case](../guide/finite_lempo.md). We can check this result using VUMPS.

```@example infinite
function gs_energy_density(; spin::Real = 1/2, ga::Real = 1.0)
    H = su2_full(; spin, ga)

    virtual_space = [SU2Space([i => 10 for i=0:ceil(spin)]...), SU2Space([i => 10 for i=(isinteger(spin) ? (ceil(spin/2):spin) : (1/2:1/2:spin))]...)]
    ψ₀ = InfiniteMPS(physicalspace(H), virtual_space)

    ψ, _, _ = find_groundstate(ψ₀, H, VUMPS())

    return real(expectation_value(ψ, H))
end

function limit_density(spin::Real = 1/2)
    j2 = round(2*spin)
    if mod(j2, 4) == 0
        (spin/2)*(spin/2 + 1)/2
    elseif mod(j2, 4) == 2
        (spin + 1)^2/8
    else
        (spin^2 + 2spin + 1/4)/8
    end
end

results = []
gas = 4:4:20
spins = reverse(collect(1/2:1/2:3))
for spin in spins
    spin_results = []
    for ga in gas
        ε = gs_energy_density(; spin, ga)
        push!(spin_results, ε/ga)
    end
    push!(results, spin_results)
end

using Plots, LaTeXStrings

plot(; xlabel=L"1/ga", ylabel=L"\varepsilon_0/g^2a", legend=:topright, grid=true)
colors = [:blue, :red, :green, :purple, :orange, :brown]
for (i, spin) in enumerate(spins)
    plot!(1 ./ gas, results[i]; label="spin = $spin", color=colors[i], marker=:circle)
end
hline!([limit_density(spin) for spin in spins]; linestyle=:dash, color=:gray, label="")
```