module OneTwoTree


#Main Module file
export tree_prediction, Node, DecisionTree, build_tree, fit!, print_tree
export lessThanOrEqual, equal
export gini_impurity

include("Tree.jl")
include("CARTutils.jl")
include("Gini.jl")


end # end the module


#Test