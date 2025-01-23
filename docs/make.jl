using OneTwoTree
using Documenter

DocMeta.setdocmeta!(OneTwoTree, :DocTestSetup, :(using OneTwoTree); recursive=true)

makedocs(;
    modules=[OneTwoTree],
    authors="Jakob Balasus <balasus@campus.tu-berlin.de>, Eloi Sandt <eloi.sandt@campus.tu-berlin.de>, Alexander Obradovic <obradovic@campus.tu-berlin.de>",
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
    ],
)

deploydocs(;
    repo="github.com/nichtJakob/OneTwoTree.jl",
    devbranch="master",
)
