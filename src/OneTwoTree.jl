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

# Decision Tree and Random Forest API
export DecisionTreeClassifier, DecisionTreeRegressor, AbstractDecisionTree
export ForestClassifier, ForestRegressor, AbstractForest, print_forest
export fit!, predict
export calc_accuracy, print_tree

# Splitting Criteria
export gini_impurity
export information_gain
export var_gain
export less_than_or_equal, equal

end # end the module