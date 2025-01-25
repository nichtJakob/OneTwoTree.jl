using OneTwoTree
using Documenter

DocMeta.setdocmeta!(OneTwoTree, :DocTestSetup, :(using OneTwoTree); recursive=true)

makedocs(;
    modules=[OneTwoTree],
    sitename="OneTwoTree.jl",
    format=Documenter.HTML(;
        canonical="https://nichtJakob.github.io/OneTwoTree.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Functions" => "functions.md",
        "Demo classifications" => "demo_classification.md",
        "Demo regression" => "demo_regression.md",
    ],
)

deploydocs(;
    repo="github.com/nichtJakob/OneTwoTree.jl",
    devbranch="master",
)
