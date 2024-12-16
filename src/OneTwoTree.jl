module OneTwoTree

#Main Module file
include("utils/load_data.jl")
include("Tree.jl")
include("CART.jl")
include("CARTutils.jl")
include("Gini.jl")


# Public API
export Node, DecisionTreeClassifier, DecisionTreeRegressor, AbstractDecisionTree
export fit!, predict, print_tree

#TODO: add build_tree guards to fit since we only export fit

# Private Utilities
export lessThanOrEqual, equal
export load_data
export gini_impurity

# Testing
export calc_depth, calc_accuracy, is_leaf

end # end the module

