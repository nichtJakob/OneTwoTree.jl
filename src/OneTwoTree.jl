module OneTwoTree

#Main Module file
include("utils/load_data.jl")
include("Node.jl")
include("Tree.jl")
include("CART.jl")
include("CARTutils.jl")
include("Gini.jl")
include("Forest.jl")
include("InfoGain.jl")
include("VarGain.jl")
include("vectorutils.jl")

export DecisionTreeClassifier, DecisionTreeRegressor, AbstractDecisionTree
export fit!, predict, print_tree, get_random_features
export calc_accuracy, print_tree
export ForestClassifier, fit2!, predict2
export gini_impurity
export information_gain
export less_than_or_equal, equal
export var_gain

end # end the module