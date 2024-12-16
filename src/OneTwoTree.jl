module OneTwoTree

#Main Module file
include("utils/load_data.jl")
include("Tree.jl")
include("CART.jl")
include("CARTutils.jl")
include("Gini.jl")

export Node, DecisionTreeClassifier, DecisionTreeRegressor, AbstractDecisionTree
export predict, fit!, build_tree, print_tree

# Private Utilities
export lessThanOrEqual, equal
export load_data
export gini_impurity

# Testing
export calc_depth, calc_accuracy, is_leaf

end # end the module

