using OneTwoTree
using Test

@testset "Gini Impurity" begin
    # Test 1: Integer labels
    @testset "Test with Integer Labels" begin
        features1 = collect(transpose([1 2 3 4]))  # Integer features
        labels1 = [1, 0, 1, 0]    # Integer labels
        node_data1 = [1, 2, 3, 4]  # All elements included

        split_true, split_false = OneTwoTree.split_indices(features1, node_data1, OneTwoTree.less_than_or_equal, 2.0, 1)
        gini1 = OneTwoTree.gini_impurity(view(labels1, split_true), view(labels1, split_false))
        @test isapprox(gini1, 0.5, atol=1e-2)  # Expected result
    end

    # Test 2: String labels
    @testset "Test with String Labels" begin
        features2 = reshape(["high", "low", "medium", "high"], 4, 1)
        labels2 = ["yes", "no", "yes", "no"]  # String labels
        node_data2 = [1, 2, 3, 4]

        split_true, split_false = OneTwoTree.split_indices(features2, node_data2, OneTwoTree.equal, "high", 1)
        gini2 = OneTwoTree.gini_impurity(view(labels2, split_true), view(labels2, split_false))
        @test isapprox(gini2, 0.5, atol=1e-2)  # Expected result
    end

    # Test 3: Multi-class String labels (new test case)
    @testset "Test with Multi-Class String Labels" begin
        features3 = reshape(["small" "medium" "large" "medium" "small" "large"], 6, 1)
        labels3 = ["low", "medium", "high", "medium", "low", "high"]  # Multi-class labels
        node_data3 = [1, 2, 3, 4, 5, 6]  # All elements included

        split_true, split_false = OneTwoTree.split_indices(features3, node_data3, OneTwoTree.equal, "medium", 1)
        gini3 = OneTwoTree.gini_impurity(view(labels3, split_true), view(labels3, split_false))
        @test isapprox(gini3, 0.333, atol=1e-2)  # Expected gini value for multi-class split
    end

    # Test 4: All labels are the same
    @testset "Test 5: All labels are the same" begin
        features5 = reshape([1, 2, 3, 4, 5], 5, 1)
        labels5 = [true, true, true, true, true]  # All labels are 'true'
        node_data5 = [1, 2, 3, 4, 5]  # All elements included

        split_true, split_false = OneTwoTree.split_indices(features5, node_data5, OneTwoTree.less_than_or_equal, 3.0, 1)
        gini5 = OneTwoTree.gini_impurity(view(labels5, split_true), view(labels5, split_false))
        @test gini5 == 0.0  # Expect 0 because all labels are the same
    end

    # Test 5: Perfect split
    @testset "Test 6: Perfect split" begin
        features6 = reshape([1, 2, 3, 4, 5, 6], 6, 1)
        labels6 = [true, true, true, false, false, false]  # Labels perfectly split
        node_data6 = [1, 2, 3, 4, 5, 6]  # All elements included

        split_true, split_false = OneTwoTree.split_indices(features6, node_data6, OneTwoTree.less_than_or_equal, 3.0, 1)
        gini6 = OneTwoTree.gini_impurity(view(labels6, split_true), view(labels6, split_false))
        @test gini6 == 0.0  # Expect 0 because it's a perfect split
    end

    # Test 6: Uneven split with imbalance
    @testset "Test 7: Uneven split with imbalance" begin
        features7 = reshape([10, 20, 30, 40, 50], 5, 1)
        labels7 = [true, true, false, false, false]  # Uneven split
        node_data7 = [1, 2, 3, 4, 5]  # All elements included

        split_true, split_false = OneTwoTree.split_indices(features7, node_data7, OneTwoTree.less_than_or_equal, 35.0, 1)
        gini7 = OneTwoTree.gini_impurity(view(labels7, split_true), view(labels7, split_false))
        @test isapprox(gini7, 0.266, atol=1e-2)  # Expected gini value with this split
    end

    # Test 7: Subset of node_data
    @testset "Test 8: Subset of node_data" begin
        features8 = reshape(["high", "low", "medium", "high"], 4, 1)
        labels8 = [true, false, true, false]
        node_data8 = [1, 2, 3]  # Only first three elements

        split_true, split_false = OneTwoTree.split_indices(features8, node_data8, OneTwoTree.equal, "high", 1)
        gini8 = OneTwoTree.gini_impurity(view(labels8, split_true), view(labels8, split_false))
        @test isapprox(gini8, 0.333, atol=1e-2)  # Expected gini value for this subset
    end

    # Test 8: No matching decision function
    @testset "Test 9: No matching decision function" begin
        features9 = reshape(["medium", "low", "medium", "low"], 4, 1)
        labels9 = [true, false, true, false]
        node_data9 = [1, 2, 3, 4]  # All elements included

        split_true, split_false = OneTwoTree.split_indices(features9, node_data9, OneTwoTree.equal, "high", 1)
        gini9 = OneTwoTree.gini_impurity(view(labels9, split_true), view(labels9, split_false))
        @test gini9 == 0.5
    end

    @testset "Empty returns 0 and error returns -1" begin
        @test OneTwoTree.gini_impurity([]) == 0
        @test OneTwoTree.gini_impurity([], []) == -1
    end
end