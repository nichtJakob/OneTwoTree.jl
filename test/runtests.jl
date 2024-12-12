using OneTwoTree
using Test

function get_test_Tree_less_0_5() # returns 1.0 if the input in dim 1 is less 0.5 else 0.0
    leaf1 = Node(prediction=1.0)
    leaf2 = Node(prediction=0.0)
    root = Node(decision = x -> lessThan(x, 0.5), true_child = leaf1, false_child = leaf2)
    return root
end


@testset "Tree.jl" begin # Tests the functionality of Node, tree_prediction, less in Tree.jl
    @testset "Tree Prediction" begin
        root = get_test_Tree_less_0_5()
        @test tree_prediction(root, [1.0]) == 0.0
        @test tree_prediction(root, [0.0]) == 1.0
        @test tree_prediction(root, [55.0]) == 0.0
        @test tree_prediction(root, [-1.0]) == 1.0
    end

    include("decision_tree_tests.jl")
end

include("gini_tests.jl")

