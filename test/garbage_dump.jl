### CART Debugging examples/tests

dataset = [[3,4] [6,1] [0,2]]
cat_labels = ["Chicken", "Egg"]
reg_labels = [12.0, 4.5]

# root = Node(dataset, cat_labels, true)
# root = Node(dataset, reg_labels, false)

# In this form the inner arrays are interpreted as column vectors of the matrix, not row vectors!
# But our implementation wants the datapoints to be the row vectors, which is meh
# So for [[] [] []] you have to set column_data to true when creating a node
dataset1 = [[3.5,1.0,5.6] [9.1,1.2,3.3] [2.9,0.4,4.3]]
dataset2 = [["Snow","Lax","Arm"] ["Hard","Snow","Hard"] ["Arm","Page","Payoff"]]
dataset3 = [[3.1,"Lax","Arm"] [0.6,"Snow","Hard"] [4.2,"Page","Payoff"]]
dataset4 = [[7.1, 3.4, 3.2, 1.8, 8.0, 0.2] [0.6, 4.3, 2.1, 0.3, 9.2, 6.3] [4.2, 4.2, 3.3, 7.4, 2.3, 2.3] [6.4, 3.2, 6.6, 4.6, 2.1, 4.1] [4.6, 0.0, 0.4, 3.6, 2.1, 0.4] [5.3, 1.3, 5.3, 2.5, 1.3, 3.2]]
dataset5 = [[7.1, 3.4, 3.2, 1.8, 8.0] [0.6, 4.3, 2.1, 0.3, 9.2] [4.2, 4.2, 3.3, 7.4, 2.3] [6.4, 3.2, 6.6, 4.6, 2.1] [4.6, 0.0, 0.4, 3.6, 2.1] [5.3, 1.3, 5.3, 2.5, 1.3]]
dataset6 = [7.1 3.4 3.2 1.8 8.0; 0.6 4.3 2.1 0.3 9.2; 4.2 4.2 3.3 7.4 2.3; 6.4 3.2 6.6 4.6 2.1; 4.6 0.0 0.4 3.6 2.1; 5.3 1.3 5.3 2.5 1.3]
cat_labels1 = ["Chicken", "Egg", "Egg"]
cat_labels2 = ["Günther Jauch", "Jürgen Jürgensson", "Sambal Oelek", "Islam Makhachev", "Gerhard Schröder", "Mein Stepptanzlehrer"]
reg_labels1 = [12.0, 4.5, 6.7]
reg_labels2 = [1.0, -4.5, 8.7, 3.2, 4.1, 6.2]

# root = Node(dataset1, cat_labels1, true)
# root = Node(dataset1, reg_labels1, false)
# root = Node(dataset1, cat_labels1, true, max_depth=1)
# root = Node(dataset1, reg_labels1, false, max_depth=1)
# root = Node(dataset2, cat_labels1, true, max_depth=1)
# root = Node(dataset2, reg_labels1, false, max_depth=1)
root = Node(dataset5, cat_labels2, true, max_depth=3, column_data=true)
# root = Node(dataset6, cat_labels2, true, max_depth=1)
tree = DecisionTree(root, 3)
print_tree(tree)
tree = build_tree(dataset5, cat_labels2, 3, column_data=true)
print_tree(tree)
fit!(tree, dataset6, reg_labels2, 3)
print_tree(tree)