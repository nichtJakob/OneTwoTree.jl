
using OneTwoTree
using Test

@testset "Gini Impurity Tests with Non-Boolean Labels" begin
    # Test 1: Integer labels
    @testset "Test with Integer Labels" begin
        features1 = [1, 2, 3, 4]  # Integer features
        labels1 = [1, 0, 1, 0]    # Integer labels
        node_data1 = [1, 2, 3, 4]  # All elements included
        decision_fn1 = x -> x <= 2
        gini1 = gini_impurity(features1, labels1, node_data1, decision_fn1)
        @test isapprox(gini1, 0.5, atol=1e-2)  # Expected result
    end

    # Test 2: String labels
    @testset "Test with String Labels" begin
        features2 = ["high", "low", "medium", "high"]
        labels2 = ["yes", "no", "yes", "no"]  # String labels
        node_data2 = [1, 2, 3, 4]
        decision_fn2 = x -> x == "high"
        gini2 = gini_impurity(features2, labels2, node_data2, decision_fn2)
        @test isapprox(gini2, 0.5, atol=1e-2)  # Expected result
    end

    # Test 3: Multi-class String labels (new test case)
    @testset "Test with Multi-Class String Labels" begin
        features3 = ["small", "medium", "large", "medium", "small", "large"]
        labels3 = ["low", "medium", "high", "medium", "low", "high"]  # Multi-class labels
        node_data3 = [1, 2, 3, 4, 5, 6]  # All elements included
        decision_fn3 = x -> x == "medium"  # Split on "medium"
        gini3 = gini_impurity(features3, labels3, node_data3, decision_fn3)
        @test isapprox(gini3, 0.333, atol=1e-2)  # Expected gini value for multi-class split
    end

    # Test 4: Empty features and labels
    @testset "Test 4: Empty features and labels" begin
        features4 = []  # Empty features
        labels4 = []    # Empty labels
        node_data4 = Int[]  # No elements
        decision_fn4 = x -> x > 30
        gini4 = gini_impurity(features4, labels4, node_data4, decision_fn4)
        @test gini4 == 0.0  # Expect 0 because there are no elements
    end

    # Test 5: All labels are the same
    @testset "Test 5: All labels are the same" begin
        features5 = [1, 2, 3, 4, 5]
        labels5 = [true, true, true, true, true]  # All labels are 'true'
        node_data5 = [1, 2, 3, 4, 5]  # All elements included
        decision_fn5 = x -> x > 3
        gini5 = gini_impurity(features5, labels5, node_data5, decision_fn5)
        @test gini5 == 0.0  # Expect 0 because all labels are the same
    end

    # Test 6: Perfect split
    @testset "Test 6: Perfect split" begin
        features6 = [1, 2, 3, 4, 5, 6]
        labels6 = [true, true, true, false, false, false]  # Labels perfectly split
        node_data6 = [1, 2, 3, 4, 5, 6]  # All elements included
        decision_fn6 = x -> x <= 3
        gini6 = gini_impurity(features6, labels6, node_data6, decision_fn6)
        @test gini6 == 0.0  # Expect 0 because it's a perfect split
    end

    # Test 7: Uneven split with imbalance
    @testset "Test 7: Uneven split with imbalance" begin
        features7 = [10, 20, 30, 40, 50]
        labels7 = [true, true, false, false, false]  # Uneven split
        node_data7 = [1, 2, 3, 4, 5]  # All elements included
        decision_fn7 = x -> x < 35
        gini7 = gini_impurity(features7, labels7, node_data7, decision_fn7)
        @test isapprox(gini7, 0.266, atol=1e-2)  # Expected gini value with this split
    end

    # Test 8: Subset of node_data
    @testset "Test 8: Subset of node_data" begin
        features8 = ["high", "low", "medium", "high"]
        labels8 = [true, false, true, false]
        node_data8 = [1, 2, 3]  # Only first three elements
        decision_fn8 = x -> x == "high"
        gini8 = gini_impurity(features8, labels8, node_data8, decision_fn8)
        @test isapprox(gini8, 0.333, atol=1e-2)  # Expected gini value for this subset
    end

    # Test 9: No matching decision function
    @testset "Test 9: No matching decision function" begin
        features9 = ["medium", "low", "medium", "low"]
        labels9 = [true, false, true, false]
        node_data9 = [1, 2, 3, 4]  # All elements included
        decision_fn9 = x -> x == "high"  # No match in decision_fn
        gini9 = gini_impurity(features9, labels9, node_data9, decision_fn9)
        @test gini9 == 0.0  # Expect 0 because no labels match the decision function
    end
end