### Main file containing all the available tests
### execute `test` in Pkg> mode (by pressing `]` in julia REPL)
### or in your julia REPL execute `include("test/runtests.jl")`

using OneTwoTree
using Test

include("data_tests.jl")
include("trees_tests/decision_tree_tests.jl")
include("trees_tests/cart_tests.jl")
include("trees_tests/regression_tests.jl")
include("trees_tests/cart_utils_tests.jl")
include("trees_tests/forest_tests.jl")
include("splitting_criteria_tests/gini_tests.jl")
include("splitting_criteria_tests/info_gain_tests.jl")
include("splitting_criteria_tests/var_gain_tests.jl")

