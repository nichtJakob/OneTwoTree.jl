### Main file containing all the available tests
### execute `test` in Pkg> mode (by pressing `]` in julia REPL)
### or in your julia REPL execute `include("test/runtests.jl")`

using OneTwoTree
using Test

include("data_tests.jl")
include("decision_tree_tests.jl")
include("gini_tests.jl")
include("cart_tests.jl")
include("regression_tests.jl")
include("cart_utils_tests.jl")

