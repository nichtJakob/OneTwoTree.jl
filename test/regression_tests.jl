### Test DecisionTrees for Regression tasks

using Test
using OneTwoTree

@testset "DecisionTreeRegressor" begin
    #Data
    r1_features = [1.0 2.0; 2.0 3.0; 3.0 4.0; 4.0 5.0]
    r1_labels = [1.5, 2.5, 3.5, 4.5]
    r1_test_features = [1.5 2.5; 3.5 4.5]

    @testset "Regression positiv" begin

    #Tree generation
    r1_tree = DecisionTreeRegressor(max_depth=3)
    fit!(r1_tree, r1_features, r1_labels)

    #predicting
    r1_predictions = predict(r1_tree, r1_test_features)

    #print_tree(r1_tree)
    @test all(isapprox.(r1_predictions, [1.5, 3.5], atol=0.1))
    end

    @testset "Regression negative" begin
    #Data
    r2_features = [-1.0 -2.0; -2.0 -3.0; -3.0 -4.0; -4.0 -5.0]
    r2_labels = [-1.5, -2.5, -3.5, -4.5]
    r2_test_features = [-1.5 -2.5; -3.5 -4.5]

    #Tree generation
    r2_tree = DecisionTreeRegressor(max_depth=3)
    fit!(r2_tree, r2_features, r2_labels)

    #predicting
    r2_predictions = predict(r2_tree, r2_test_features)

    #print_tree(r2_tree)
    @test all(isapprox.(r2_predictions, [-2.5, -4.5], atol=0.1))
    end

    @testset "Regression depth 1" begin

    #Tree generation
    r3_tree = DecisionTreeRegressor(max_depth=1)
    fit!(r3_tree, r1_features, r1_labels)

    #predicting
    r3_predictions = predict(r3_tree, r1_test_features)
    #print(r3_predictions)

    #print_tree(r3_tree)
    @test all(isapprox.(r3_predictions, [1.5, 3.5], atol=0.1))
    end

    @testset "Regression max depth not set" begin

        #Tree generation
        r4_tree = DecisionTreeRegressor()
        fit!(r4_tree, r1_features, r1_labels)

        #predicting
        r4_predictions = predict(r4_tree, r1_test_features)
        #print(r4_predictions)

        #print_tree(r4_tree)
        @test all(isapprox.(r4_predictions, [1.5, 3.5], atol=0.1))
    end

    @testset "Regression max depth = 100 000" begin

        #Tree generation
        tree = DecisionTreeRegressor(max_depth = 100000)
        fit!(tree, r1_features, r1_labels)

        #predicting
        predictions = predict(tree, r1_test_features)
        #print(predictions)

        #print_tree(tree)
        @test all(isapprox.(predictions, [1.5, 3.5], atol=0.1))
    end
end
