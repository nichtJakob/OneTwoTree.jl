using OneTwoTree
using Documenter

DocMeta.setdocmeta!(OneTwoTree, :DocTestSetup, :(using OneTwoTree); recursive=true)

makedocs(;
    checkdocs = :none,
    modules=[OneTwoTree],
    sitename="OneTwoTree.jl",
    authors="Jakob Balasus <balasus@campus.tu-berlin.de>, Eloi Sandt <eloi.sandt@campus.tu-berlin.de>, Andreas Paul Bruno Lönne <loenne@campus.tu-berlin.de>, Alexander Obradovic <obradovic@campus.tu-berlin.de>",
    format=Documenter.HTML(;
        canonical="https://nichtJakob.github.io/OneTwoTree.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Functions" => "functions.md",
        "Demo Classification" => "demo_classification.md",
        "Demo Regression" => "demo_regression.md",
    ],
)

deploydocs(;
    repo="github.com/nichtJakob/OneTwoTree.jl",
    devbranch="master",
)
