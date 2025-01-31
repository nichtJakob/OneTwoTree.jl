# Run the demo examples of the ReadMe and external documentation as tests to make sure
# they are working at all times.

using Test
using OneTwoTree
using Suppressor # suppress prints in tests
using MLDatasets: Iris, BostonHousing
using DataFrames
using Statistics
using Random

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

@testset "ReadMe Examples" begin
    @testset "Classification" begin
        dataset = [
            3.5 9.1 2.9
            1.0 1.2 0.4
            5.6 3.3 4.3
        ]
        labels = ["A", "B", "C"]

        tree = DecisionTreeClassifier(max_depth=2)
        @capture_out print(tree)
        @capture_out print(fit!(tree, dataset, labels))
        @capture_out print(tree)
        prediction = predict(tree, [
            2.0 4.0 6.0
        ])
        @capture_out print("The tree predicted class $(prediction[1]).")
        @test true # no errors thrown
    end

    @testset "Regression" begin
        dataset = [
            1.0 2.0
            2.0 3.0
            3.0 4.0
            4.0 5.0
        ]
        labels = [1.5, 2.5, 3.5, 4.5]

        tree = DecisionTreeRegressor(max_depth=3)
        @capture_out print(tree)
        @capture_out print(fit!(tree, dataset, labels))
        @capture_out print(tree)

        prediction = predict(tree, [
            1.0 4.0
        ])
        @capture_out print("The tree predicted $(prediction[1]).")
        @test true # no errors thrown
    end
end

@testset "Iris Demo Example" begin
    dataset = Iris()
    data = Array(dataset.features)
    targets = String.(Array(dataset.targets))

    @capture_out println("The possible targets are: ", unique(targets), "\n")
    @capture_out println("The measured features are: ", names(dataset.features), "\n")
    @capture_out println("Size of data: ", size(data), "\n")

    splitting_point = 120
    if splitting_point < 1 || splitting_point > 150
        error("You have chosen an invalid splitting point. Please choose a value between 1 and 150")
    end

    train_data = data[1:splitting_point, :]
    train_labels = targets[1:splitting_point]
    test_data = data[splitting_point+1:150, :]
    test_labels = targets[splitting_point+1:150]

    @capture_out println("Size of train data: ", size(train_data), "\n")
    @capture_out println("Size of test data: ", size(test_data), "\n")

    our_max_depth = 3
    tree = DecisionTreeClassifier(max_depth=our_max_depth)
    fit!(tree, train_data, train_labels)

    @capture_out println("\n\nOur Tree:\n")
    @capture_out print_tree(tree)

    test_predictions = predict(tree, test_data)
    accuracy = sum(test_predictions .== test_labels) / length(test_labels)
    @test accuracy > 0.0

    @capture_out println("\n\nFor the Iris dataset we have achieved a test accuracy of $(round(accuracy * 100, digits=2))%")
    @capture_out println("-----------------------------------------------------\n")

    n_features_per_tree = 30
    @capture_out println("\n\nNow we will grow our random forest containing 5 trees with $n_features_per_tree features per tree and a max depth of $our_max_depth")

    forest = ForestClassifier(n_trees=5, n_features_per_tree=n_features_per_tree, max_depth=our_max_depth)
    fit!(forest, train_data, train_labels)

    @capture_out print_forest(forest)

    forest_test_predictions = predict(forest, test_data)
    forest_accuracy = sum(forest_test_predictions .== test_labels) / length(test_labels)
    @test forest_accuracy > 0.0

    @capture_out println("\n\nFor the Iris dataset the forest has achieved a test accuracy of $(round(forest_accuracy * 100, digits=2))%")
end

@testset "BostonHousing Demo Example" begin
    dataset = BostonHousing(as_df=false)
    X, y = dataset[:]
    n_samples = size(X, 1)

    train_ratio = 0.8
    n_train = Int(round(train_ratio * n_samples))
    indices = randperm(n_samples)
    train_idx = indices[1:n_train]
    test_idx = indices[n_train+1:end]
    X_train, y_train = X[train_idx, :], y[train_idx]
    X_test, y_test = X[test_idx, :], y[test_idx]

    tree = DecisionTreeRegressor(max_depth=5)
    fit!(tree, X_train, y_train)

    @capture_out println("\n\nOur Tree:\n")
    @capture_out print_tree(tree)

    forest = ForestRegressor(n_trees=5, n_features_per_tree=40, max_depth=30)
    fit!(forest, X_train, y_train)

    @capture_out println("\n \n Our forest: \n")
    @capture_out print_forest(forest)

    y_pred_tree = predict(tree, X_test)

    mse_tree = mean((y_pred_tree .- y_test).^2)  # Mean Squared Error
    rmse_tree = sqrt(mse_tree)                  # Root Mean Squared Error
    mae_tree = mean(abs.(y_pred_tree .- y_test))  # Mean Absolute Error

    y_pred_forest = predict(forest, X_test)

    mse_forest = mean((y_pred_forest .- y_test).^2)  # Mean Squared Error
    rmse_forest = sqrt(mse_forest)                  # Root Mean Squared Error
    mae_forest = mean(abs.(y_pred_forest .- y_test))  # Mean Absolute Error

    @capture_out println("\nTest Performance comparison:")
    @capture_out println("-------------------------------\n")

    @capture_out println("Mean Squared Error (MSE)")
    @capture_out println("Tree   (MSE): $mse_tree")
    @capture_out println("Forest (MSE): $mse_forest\n")

    @capture_out println("Root Mean Squared Error (RMSE)")
    @capture_out println("Tree   (RMSE): $rmse_tree")
    @capture_out println("Forest (RMSE): $rmse_forest\n")

    @capture_out println("Mean Absolute Error (MAE)")
    @capture_out println("Tree   (MAE): $mae_tree")
    @capture_out println("Forest (MAE): $mae_forest\n")
    @test true # no errors thrown
end