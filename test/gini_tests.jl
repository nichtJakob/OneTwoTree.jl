
using OneTwoTree
using Test

@testset "Gini.jl Tests" begin
    # Test 1: Boolean features and labels
    @testset "Test 1: Boolean features and labels" begin
        features1 = Union{Real, String}[true, false, true, true, false]
        labels1 = Union{Real, String}[true, false, true, false, false]
        node_data1 = [1, 2, 3, 4, 5]  # All elements included
        decision_fn1 = x -> x == true
        gini1 = gini_impurity(features1, labels1, node_data1, decision_fn1)
        @test isapprox(gini1, 0.266, atol=1e-2)
    end

    # Test 2: Numerical features
    @testset "Test 2: Numerical features" begin
        features2 = Union{Real, String}[25, 40, 35, 22, 60]
        labels2 = Union{Real, String}[true, false, true, false, true]
        node_data2 = [1, 2, 3, 4, 5]  # All elements included
        decision_fn2 = x -> x > 30
        gini2 = gini_impurity(features2, labels2, node_data2, decision_fn2)
        @test isapprox(gini2, 0.466, atol=1e-2)
    end

    # Test 3: Empty features and labels
    @testset "Test 3: Empty features and labels" begin
        features3 = Union{Real, String}[]
        labels3 = Union{Real, String}[]
        node_data3 = Int[]  # No elements
        decision_fn3 = x -> x > 30
        gini3 = gini_impurity(features3, labels3, node_data3, decision_fn3)
        @test gini3 == 0.0
    end

    # Test 4: All labels are the same
    @testset "Test 4: All labels are the same" begin
        features4 = Union{Real, String}[1, 2, 3, 4, 5]
        labels4 = Union{Real, String}[true, true, true, true, true]
        node_data4 = [1, 2, 3, 4, 5]  # All elements included
        decision_fn4 = x -> x > 3
        gini4 = gini_impurity(features4, labels4, node_data4, decision_fn4)
        @test gini4 == 0.0
    end

    # Test 5: Perfect split
    @testset "Test 5: Perfect split" begin
        features5 = Union{Real, String}[1, 2, 3, 4, 5, 6]
        labels5 = Union{Real, String}[true, true, true, false, false, false]
        node_data5 = [1, 2, 3, 4, 5, 6]  # All elements included
        decision_fn5 = x -> x <= 3
        gini5 = gini_impurity(features5, labels5, node_data5, decision_fn5)
        @test gini5 == 0.0
    end

    # Test 6: Uneven split with imbalance
    @testset "Test 6: Uneven split with imbalance" begin
        features6 = Union{Real, String}[10, 20, 30, 40, 50]
        labels6 = Union{Real, String}[true, true, false, false, false]
        node_data6 = [1, 2, 3, 4, 5]  # All elements included
        decision_fn6 = x -> x < 35
        gini6 = gini_impurity(features6, labels6, node_data6, decision_fn6)
        @test isapprox(gini6, 0.266, atol=1e-2)
    end

    # Test 7: Subset of node_data
    @testset "Test 7: Subset of node_data" begin
        features7 = Union{Real, String}["high", "low", "medium", "high"]
        labels7 = Union{Real, String}[true, false, true, false]
        node_data7 = [1, 2, 3]  # Only first three elements
        decision_fn7 = x -> x == "high"
        gini7 = gini_impurity(features7, labels7, node_data7, decision_fn7)
        @test isapprox(gini7, 0.333, atol=1e-2)
    end

    # Test 8: No matching decision function
    @testset "Test 8: No matching decision function" begin
        features8 = Union{Real, String}["medium", "low", "medium", "low"]
        labels8 = Union{Real, String}[true, false, true, false]
        node_data8 = [1, 2, 3, 4]  # All elements included
        decision_fn8 = x -> x == "high"  # No match
        gini8 = gini_impurity(features8, labels8, node_data8, decision_fn8)
        @test gini8 == 0.0
    end
end