# MPSKitLEMPO.jl

Documentation for [MPSKitLEMPO.jl](https://github.com/benjamints/MPSKitLEMPO.jl).

## Overview

`MPSKitLEMPO.jl` is a Julia package that extends [MPSKit.jl](https://github.com/maartenvd/MPSKit.jl) to support Link-Enhanced Matrix Product Operator (LEMPO) Hamiltonians, following the construction in [this paper](https://arxiv.org/abs/2508.16363). 

## Features

Using `MPSKitLEMPO.jl`, you can take a (finite or infinite) MPO object from `MPSKit.jl` and augment it with a sum of on-site link operators. We implement the following algorithms for LEMPOs:
- DMRG
- DMRG2
- VUMPS
- QuasiparticleAnsatz

## Installation

`MPSKitLEMPO.jl` is currently under development. You can install it from the repository:

```julia
using Pkg
Pkg.add(url="https://github.com/benjamints/MPSKitLEMPO.jl")
```

Or for development:

```julia
using Pkg
Pkg.develop(url="https://github.com/benjamints/MPSKitLEMPO.jl")
```

## Table of Contents

```@contents
Pages = [
    "guide/getting_started.md",
    "guide/finite_lempo.md",
    "guide/infinite_lempo.md",
    "api/reference.md"
]
```

## Index

```@index
```
