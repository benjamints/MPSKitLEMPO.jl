using Documenter
using MPSKitLEMPO

makedocs(
    sitename = "MPSKitLEMPO.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://benjamints.github.io/MPSKitLEMPO.jl",
        assets = String[],
    ),
    modules = [MPSKitLEMPO],
    pages = [
        "Home" => "index.md",
        "User Guide" => [
            "Getting Started" => "guide/getting_started.md",
            "Finite LEMPO" => "guide/finite_lempo.md",
            "Infinite LEMPO" => "guide/infinite_lempo.md",
        ],
        "API Reference" => "api/reference.md"
    ],
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/benjamints/MPSKitLEMPO.jl.git",
    devbranch = "main",
)
