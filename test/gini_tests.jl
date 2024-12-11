using OneTwoTree
using Test

using OneTwoTree
using Test

@testset "Gini.jl Tests" begin
    # Test 1: Boolean features and labels
    @testset "Test 1: Boolean features and labels" begin
        features1 = [true, false, true, true, false]
        labels1 = [true, false, true, false, false]
        decision_fn1 = x -> x == true
        gini1 = gini_impurity(features1, labels1, decision_fn1)
        @test isapprox(gini1, 0.266, atol=1e-2)
    end

    # Test 2: Numerical features
    @testset "Test 2: Numerical features" begin
        features2 = [25, 40, 35, 22, 60]
        labels2 = [true, false, true, false, true]
        decision_fn2 = x -> x > 30
        gini2 = gini_impurity(features2, labels2, decision_fn2)
        @test isapprox(gini2, 0.466, atol=1e-2)
    end

    # Test 3: Empty features and labels
    @testset "Test 3: Empty features and labels" begin
        features3 = Int[]
        labels3 = Bool[]
        decision_fn3 = x -> x > 30
        gini3 = gini_impurity(features3, labels3, decision_fn3)
        @test gini3 == 0.0
    end

    # Test 4: All labels are the same
    @testset "Test 4: All labels are the same" begin
        features4 = [1, 2, 3, 4, 5]
        labels4 = [true, true, true, true, true]
        decision_fn4 = x -> x > 3
        gini4 = gini_impurity(features4, labels4, decision_fn4)
        @test gini4 == 0.0
    end

    # Test 5: Perfect split
    @testset "Test 5: Perfect split" begin
        features5 = [1, 2, 3, 4, 5, 6]
        labels5 = [true, true, true, false, false, false]
        decision_fn5 = x -> x <= 3
        gini5 = gini_impurity(features5, labels5, decision_fn5)
        @test gini5 == 0.0
    end

    # Test 6: Uneven split with imbalance
    @testset "Test 6: Uneven split with imbalance" begin
        features6 = [10, 20, 30, 40, 50]
        labels6 = [true, true, false, false, false]
        decision_fn6 = x -> x < 35
        gini6 = gini_impurity(features6, labels6, decision_fn6)
        @test isapprox(gini6, 0.266, atol=1e-2)
    end
end





