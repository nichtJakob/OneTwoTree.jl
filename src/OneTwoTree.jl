module OneTwoTree

include("Tree.jl")
include("utils/load_data.jl")
include("utils/download_dataset.jl")
include("Gini.jl")

#Main Module file
export lessThan, fit!, tree_prediction, Node, DecisionTree, print_tree
export calc_depth, calc_accuracy, is_leaf
export load_data, save_img_dataset_as_csv
export gini_impurity


end # end the module


#Test
