module OneTwoTree

#Main Module file
include("Tree.jl")
include("utils/load_data.jl")
include("CARTutils.jl")
include("Gini.jl")

export Node, DecisionTree
export predict, fit!, build_tree, print_tree
export calc_depth, calc_accuracy, is_leaf
export lessThanOrEqual, equal
export load_data
export gini_impurity

end # end the module

