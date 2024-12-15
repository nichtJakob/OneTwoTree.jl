using Test
using OneTwoTree

run_mnist = false

"""
    test_node_consistency(node::Node)

Checks if the properties of the node are consistent with the type of the node.
"""
function test_node_consistency(node::Node)
    if is_leaf(node)
        @test node.prediction isa Real || node.prediction isa String
        @test node.true_child === nothing
        @test node.false_child === nothing
        @test node.decision === nothing
    else
        @test node.prediction === nothing
        @test node.true_child isa Node
        @test node.false_child isa Node
        @test node.decision isa Function
    end
end

"""
    test_tree_consistency(tree, run_tests::Bool=true)

Traverses the tree and checks all properties of the tree and its nodes for consistency.
"""
function test_tree_consistency(; tree::AbstractDecisionTree, run_tests::Bool=true)
    if !run_tests
        @warn "Skipping tree consistency tests"
        return
    end

    # test tree properties
    @test tree.max_depth > 0 || tree.max_depth == -1

    if tree.root === nothing
        return
    end

    # test node integrity
    to_visit = [tree.root]
    while !isempty(to_visit)
        node = popfirst!(to_visit)

        test_node_consistency(node)

        if node.true_child !== nothing
            push!(to_visit, node.true_child)
        end

        if node.false_child !== nothing
            push!(to_visit, node.false_child)
        end
    end

    # test depth consistency
    @test calc_depth(tree) <= tree.max_depth
end

@testset "FashionMNIST-1000" begin
    if !run_mnist
        @warn "Skipping FashionMNIST tests"
        return
    end

    features, labels = load_data("fashion_mnist_1000")
    tree = DecisionTreeClassifier(max_depth=10)

    @testset "Tree Construction" begin
        fit!(tree, features, labels)

        @test tree.root isa Node
        @test tree.max_depth == 10
        test_tree_consistency(tree=tree, run_tests=tree.root !== nothing)
    end

    #@warn "Skipping prediction tests"
    if(tree.root === nothing)
        @warn "Skipping prediction tests"
        return
    end
    @testset "Prediction" begin
        pred = predict(tree, features)

        @test length(pred) == length(labels)
        @test calc_accuracy(labels, pred) > 0.2
    end
end
