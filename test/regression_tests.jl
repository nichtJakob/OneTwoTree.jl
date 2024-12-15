using Test
using OneTwoTree

@testset "DecisionTreeRegressor" begin

    #Data
    r1_features = [1.0 2.0; 2.0 3.0; 3.0 4.0; 4.0 5.0]
    r1_labels = [1.5, 2.5, 3.5, 4.5]
    r1_test_features = [1.5 2.5; 3.5 4.5]

    #Tree generation
    r1_tree = DecisionTreeRegressor(max_depth=3)
    fit!(r1_tree, r1_features, r1_labels)

    #predicting
    r1_predictions = predict(r1_tree, r1_test_features)

    @test all(isapprox.(r1_predictions, [1.5, 3.5], atol=0.1))
end
