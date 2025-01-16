module OneTwoTree

#Main Module file
include("utils/load_data.jl")
include("Node.jl")
include("Tree.jl")
include("CART.jl")
include("CARTutils.jl")
include("Gini.jl")
include("infoGain.jl")


# Public API
export DecisionTreeClassifier, DecisionTreeRegressor
export fit!, predict
export calc_accuracy, print_tree

export gini_impurity

export information_gain
export less_than_or_equal, equal

end # end the module