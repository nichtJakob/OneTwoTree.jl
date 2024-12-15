using Test
using OneTwoTree


const RUN_MNIST = false
const USE_INT_FEATURES = false

"""
    test_node_consistency(node::Node)

Checks if the properties of the node are consistent with the type of the node.
"""
function test_node_consistency(node::Node)
    if is_leaf(node)
        @test node.prediction isa Number || node.prediction isa String
        @test node.true_child === nothing
        @test node.false_child === nothing
        @test node.decision === nothing
    else
        @test node.prediction === nothing
        @test node.true_child isa Node
        @test node.false_child isa Node
        @test node.decision isa OneTwoTree.Decision
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

@testset "Basic Classification" begin
    @testset "Fit and Predict" begin
        dataset1 = [
            3.0 6.0 0.0
            4.0 1.0 2.0
        ]
        cat_labels1 = ["Chicken", "Egg"]

        t1 = DecisionTreeClassifier(max_depth=1)
        fit!(t1, dataset1, cat_labels1)

        @test t1.root isa Node
        @test t1.max_depth == 1
        test_tree_consistency(tree=t1, run_tests=t1.root !== nothing)

        pred = predict(t1, dataset1)
        @test length(pred) == length(cat_labels1)
        @test calc_accuracy(cat_labels1, pred) == 1.0
    end

    @testset "Data Types" begin
        dataset_float = [
            3.5 9.1 2.9
            1.0 1.2 0.4
            5.6 3.3 4.3
        ]
        dataset_string = [
            "Snow" "Hard" "Arm"
            "Lax" "Snow" "Page"
            "Arm" "Hard" "Payoff"
        ]
        dataset_int = [
            7 0 4
            3 4 4
            3 2 3
            1 0 7
            8 9 2
            0 6 2
        ]
        if !USE_INT_FEATURES
            dataset_int = convert(Matrix{Float64}, dataset_int)
        end

        t_float = DecisionTreeClassifier(max_depth=3)
        t_string = DecisionTreeClassifier(max_depth=3)
        t_int = DecisionTreeClassifier(max_depth=3)

        fit!(t_float, dataset_float, ["A", "B", "C"])
        fit!(t_string, dataset_string, ["A", "B", "C"])
        fit!(t_int, dataset_int, ["A", "B", "C"])

        test_tree_consistency(tree=t_float, run_tests=t_float.root !== nothing)
        test_tree_consistency(tree=t_string, run_tests=t_string.root !== nothing)
        test_tree_consistency(tree=t_int, run_tests=t_int.root !== nothing)
        @test max_depth(t_float) == 3
        @test max_depth(t_string) == 3
        @test max_depth(t_int) == 3

        pred_float = predict(t_float, dataset_float)
        pred_string = predict(t_string, dataset_string)
        pred_int = predict(t_int, dataset_int)

        @test length(pred_float) == 3
        @test length(pred_string) == 3
        @test length(pred_int) == 3
        @test calc_accuracy(["A", "B", "C"], pred_float) == 1.0
        @test calc_accuracy(["A", "B", "C"], pred_string) == 1.0
        @test calc_accuracy(["A", "B", "C"], pred_int) == 1.0
        #TODO: test invalid inputs, should throw errors
    end

    @testset "Max Depth" begin
        # unlimited depth
        # bigger depth than necessary
        # smaller depth than necessary
        # depth 0
        # depth < -1
    end
end


@testset "FashionMNIST-1000" begin
    if !RUN_MNIST
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
