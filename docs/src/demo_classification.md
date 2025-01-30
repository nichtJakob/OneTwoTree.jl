 In this example we want to distinguish between 3 types of the iris flower
 using the OneTwoTree package
using OneTwoTree

 If you want to execute this example in your julia REPL you will first need to
 1) install dependencies for the dataset:
   julia>  using Pkg
   julia>  Pkg.add("MLDatasets")
   julia>  Pkg.add("DataFrames")

 2) execute the code in your REPL:
   julia>  include("demo_classification.jl")

 First we load the iris dataset. Targets are the 3 types of the flowers
 and data contains measurements of flowers
```julia
using DataFrames
using MLDatasets: Iris

dataset = Iris()
data = Array(dataset.features)
targets = String.(Array(dataset.targets))
```

 Let's have a look at the types of the iris flower contained in targets
```julia
println("The possible targets are: ", unique(targets), "\n")
```
 and the features, aka the given measurements of flowers
```julia
println("The measured features are: ", names(dataset.features), "\n")
```
 We have 150 data points as you can see in the size of the data
```julia
println("Size of data: ", size(data), "\n")
```

 Here we define how big training and test set should be
 You can modify the splitting point to be any value between 1 and 149
```julia
splitting_point = 120

if splitting_point < 1 || splitting_point > 150
    error("You have chosen an invalid splitting point. Please choose a value between 1 and 150")
end
```
 We split the data in training and test sets
```julia
train_data = data[1:splitting_point, :]
train_labels = targets[1:splitting_point]
test_data = data[splitting_point+1:150, :]
test_labels = targets[splitting_point+1:150]

println("Size of train data: ", size(train_data), "\n")
println("Size of test data: ", size(test_data), "\n")
```
 Now we use the OneTwoTree package to build a decision tree, you can experiment with different tree-depths
```julia
our_max_depth = 3
tree = DecisionTreeClassifier(max_depth=our_max_depth)
```
 Now we train on the training data
```julia
fit!(tree, train_data, train_labels)
```
 Finally, we can take a look at our trained tree
```julia
println("\n\nOur Tree:\n")
print_tree(tree)
```
 Let's see how good our tree is at predicting labels of data points in the test_data
```julia
test_predictions = predict(tree, test_data)
accuracy = sum(test_predictions .== test_labels) / length(test_labels)

println("\n\nFor the Iris dataset we have achieved a test accuracy of $(round(accuracy * 100, digits=2))%")
println("-----------------------------------------------------\n")
```
 Now let's try using random forests

 You can play around with the following values of the parameters:
 n_trees := number of trees in our forest
 n_features_per_tree := number of features per tree
 max_depth := our max depth for the trees
```julia
n_features_per_tree = 30
println("\n\nNow we will grow our random forest containing 5 trees with $n_features_per_tree features per tree and a max depth of $our_max_depth")

forest = ForestClassifier(n_trees=5, n_features_per_tree=n_features_per_tree, max_depth=our_max_depth)
fit!(forest, train_data, train_labels)
```
 Let's have a look at our forest
```julia
print_forest(forest)
```
 Let's see how good our tree is at predicting lables of datapoints in the test_data
```julia
forest_test_predictions = predict(forest, test_data)
forest_accuracy = sum(forest_test_predictions .== test_labels) / length(test_labels)

println("\n\nFor the Iris dataset the forest has achieved a test accuracy of $(round(forest_accuracy * 100, digits=2))%")
```