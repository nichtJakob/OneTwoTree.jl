  In this example we want to compare the performance
  of regression trees with regression random forests
  on the BostonHousing dataset by using the OneTwoTree package
```julia
using OneTwoTree
```
  If you want to execute this example in your julia REPL you will first need to
  1) install dependencies for the dataset:
    julia>  using Pkg
    julia>  Pkg.add("MLDatasets")
    julia>  Pkg.add("DataFrames")
 
  2) execute the code in your REPL:
    julia>  include("demo_regression.jl")
```julia
using MLDatasets: BostonHousing
using Random
using Statistics
using DataFrames
```
  First we load the data
```julia
dataset = BostonHousing(as_df=false)
X, y = dataset[:]
n_samples = size(X, 1)
```
  We split randomly into training and test sets. X are the features and y are the targets
  You can experiment with different train ratios as long as it stays between 0.1 and 0.9
```julia
train_ratio = 0.8
n_train = Int(round(train_ratio * n_samples))
indices = randperm(n_samples)
train_idx = indices[1:n_train]
test_idx = indices[n_train+1:end]
X_train, y_train = X[train_idx, :], y[train_idx]
X_test, y_test = X[test_idx, :], y[test_idx]
```
  Now we use the OneTwoTree package to plant a regression tree.
  You can experiment with different maximum tree depths.
```julia
tree = DecisionTreeRegressor(max_depth=5)
```
  We train it on the training data
```julia
fit!(tree, X_train, y_train)
```
  Let's have a look at our tree
```julia
println("\n\nOur Tree:\n")
print_tree(tree)
```
  Now let's look at regression forests
  You can experiment with the parameters and see how the performance varies
```julia
forest = ForestRegressor(n_trees=5, n_features_per_tree=40, max_depth=30)
fit!(forest, X_train, y_train)
```
  Let's admire our forest by printing it to the console
  If you have chosen a large number of trees, you might want to comment out the forest printing
```julia
println("\n \n Our forest: \n")
print_forest(forest)
```

  Let's check the tree performance on test data
```julia
y_pred_tree = predict(tree, X_test)

mse_tree = mean((y_pred_tree .- y_test).^2)    #Mean Squared Error
rmse_tree = sqrt(mse_tree)                    #Root Mean Squared Error
mae_tree = mean(abs.(y_pred_tree .- y_test))    #Mean Absolute Error
```

  And now the forest performance on test data
```julia
y_pred_forest = predict(forest, X_test)

mse_forest = mean((y_pred_forest .- y_test).^2)    #Mean Squared Error
rmse_forest = sqrt(mse_forest)                  # Root Mean Squared Error
mae_forest = mean(abs.(y_pred_forest .- y_test))  # Mean Absolute Error
```


  Finally, we can compare our regression tree with the random forest regressor
```julia
println("\nTest Performance comparison:")
println("-------------------------------\n")

println("Mean Squared Error (MSE)")
println("Tree   (MSE): $mse_tree")
println("Forest (MSE): $mse_forest\n")

println("Root Mean Squared Error (RMSE)")
println("Tree   (RMSE): $rmse_tree")
println("Forest (RMSE): $rmse_forest\n")

println("Mean Absolute Error (MAE)")
println("Tree   (MAE): $mae_tree")
println("Forest (MAE): $mae_forest\n")
```