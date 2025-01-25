module OneTwoTree

#Main Module file
include("utils/vector_utils.jl")
include("splitting_criteria/gini.jl")
include("splitting_criteria/info_gain.jl")
include("splitting_criteria/var_gain.jl")

include("trees/decision_function.jl")
include("trees/node.jl")
include("trees/tree.jl")
include("trees/forest.jl")
include("trees/cart/cart.jl")
include("trees/cart/cart_utils.jl")


# Decision Tree and Random Forest API
export DecisionTreeClassifier, DecisionTreeRegressor
export ForestClassifier, ForestRegressor
export fit!, predict
export calc_accuracy, print_tree, print_forest

# Splitting Criteria
export gini_impurity
export information_gain
export variance_gain
export less_than_or_equal, equal

end # end the module
