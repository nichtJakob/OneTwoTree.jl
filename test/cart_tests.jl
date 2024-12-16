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
    if tree.max_depth > 0
        @test calc_depth(tree) <= tree.max_depth
    end
end

@testset "Basic Classification" begin
    # some datasets
    dataset1 = [
        3.0 6.0 0.0
        4.0 1.0 2.0
    ]
    cat_labels1 = ["Chicken", "Egg"]

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
    abc_labels = ["A", "B", "C"]
    aabcbb_labels = ["A", "A", "B", "C", "B", "B"]
    @testset "Fit and Predict" begin
        t1 = DecisionTreeClassifier(max_depth=1)
        fit!(t1, dataset1, cat_labels1)

        @test t1.root isa Node
        @test t1.max_depth == 1
        test_tree_consistency(tree=t1, run_tests=t1.root !== nothing)

        pred = predict(t1, dataset1)
        @test length(pred) == length(cat_labels1)
        @test calc_accuracy(cat_labels1, pred) == 1.0

        @testset "Prediction Threshold" begin
            t1_2 = DecisionTreeClassifier(max_depth = 1)
            fit!(t1_2, reshape([
                1.0
                10.0
            ], 2, 1), ["A", "B"])
            @test predict(t1_2, [5.5]) == "A"
            @test predict(t1_2, [3.0]) == "A"
            @test predict(t1_2, [5.6]) == "B"
            @test predict(t1_2, [-200.0]) == "A"

            t1_3 = DecisionTreeClassifier(max_depth = 2)
            fit!(t1_3, reshape([
                10.0
                1.0
                3.0
            ], 3, 1), ["C", "A", "B"])
            @test predict(t1_3, [2.0]) == "A"
            @test predict(t1_3, [5.0]) == "B"
            @test predict(t1_3, [11.0]) == "C"
            @test predict(t1_3, [0.9]) == "A"
            @test predict(t1_3, [6.5005]) == "C"
        end
    end

    # TODO: readd integer and string type tests as soon as they can be handled!
    @testset "Data Types" begin
        if !USE_INT_FEATURES
            dataset_int = convert(Matrix{Float64}, dataset_int)
        end

        t_float = DecisionTreeClassifier(max_depth=3)
        t_string = DecisionTreeClassifier(max_depth=3)
        t_int = DecisionTreeClassifier(max_depth=3)

        fit!(t_float, dataset_float, abc_labels)
        fit!(t_string, dataset_string,  abc_labels)
        fit!(t_int, dataset_int, aabcbb_labels)

        @test t_float.root isa Node
        @test t_string.root isa Node
        @test t_int.root isa Node
        test_tree_consistency(tree=t_float, run_tests=t_float.root !== nothing)
        # test_tree_consistency(tree=t_string, run_tests=t_string.root !== nothing)
        # test_tree_consistency(tree=t_int, run_tests=t_int.root !== nothing)
        @test calc_depth(t_float) == 2
        # @test calc_depth(t_string) == 2
        # @test calc_depth(t_int) == 2 # TODO: this probably isn't 2 as we have 6 data points

        pred_float = predict(t_float, dataset_float)
        # pred_string = predict(t_string, dataset_string)
        # pred_int = predict(t_int, dataset_int)

        @test length(pred_float) == 3
        # @test length(pred_string) == 3
        # @test length(pred_int) == 6
        @test calc_accuracy(abc_labels, pred_float) == 1.0
        # @test calc_accuracy(abc_labels, pred_string) == 1.0
        # @test calc_accuracy(aabcbb_labels, pred_int) == 1.0

        #TODO: test mixed type and int features
        #TODO: test invalid inputs, should throw errors
    end

    # TODO: readd integer label type tests as soon as it is fixed!
    @testset "Int Label" begin
        # @warn "Int Labels are allowed in the tree code but not sure if this will work."
        # t_int_label = DecisionTreeClassifier(max_depth=3)
        # fit!(t_int_label, dataset_float, [1, 2, 3])

        # @test t_int_label.root isa Node
        # test_tree_consistency(tree=t_int_label, run_tests=t_int_label.root !== nothing)
        # @test calc_depth(t_int_label) == 3

        # pred_int_label = predict(t_int_label, dataset_float)
        # @test length(pred_int_label) == 3
        # @test calc_accuracy([1, 2, 3], pred_int_label) == 1.0
    end

    @testset "Max Depth" begin
        @testset "Zero Depth" begin
            t_zero_depth = DecisionTreeClassifier(max_depth=0)
            fit!(t_zero_depth, dataset_float, abc_labels)
            @test calc_depth(t_zero_depth) == 0
        end

        # unlimited depth
        @testset "Unlimited Depth" begin
            t_unlimited = DecisionTreeClassifier(max_depth=-1)
            fit!(t_unlimited, dataset_float, abc_labels)

            @test t_unlimited.root isa Node
            test_tree_consistency(tree=t_unlimited, run_tests=t_unlimited.root !== nothing)
            @test calc_depth(t_unlimited) > 1

            pred_unlimited = predict(t_unlimited, dataset_float)
            @test length(pred_unlimited) == length(abc_labels)
            @test calc_accuracy(abc_labels, pred_unlimited) == 1.0
        end

        @testset "Bigger Depth Than Necessary" begin
            t_bigger = DecisionTreeClassifier(max_depth=12)
            fit!(t_bigger, dataset_float, abc_labels)

            @test t_bigger.root isa Node
            test_tree_consistency(tree=t_bigger, run_tests=t_bigger.root !== nothing)
            @test calc_depth(t_bigger) <= 2

            pred_bigger = predict(t_bigger, dataset_float)
            @test length(pred_bigger) == length(abc_labels)
            @test calc_accuracy(abc_labels, pred_bigger) == 1.0
        end

        @testset "Smaller Depth Than Necessary" begin
            t_smaller = DecisionTreeClassifier(max_depth=1)
            fit!(t_smaller, dataset_float, abc_labels)

            @test t_smaller.root isa Node
            test_tree_consistency(tree=t_smaller, run_tests=t_smaller.root !== nothing)
            @test calc_depth(t_smaller) == 1

            pred_smaller = predict(t_smaller, dataset_float)
            @test length(pred_smaller) == length(abc_labels)
            @test calc_accuracy(abc_labels, pred_smaller) > 0.5
        end

        @testset "Invalid Depth" begin
            @test_throws "DecisionTreeClassifier: Got invalid max_depth. Set it to a value >= -1. (-1 means unlimited depth)" DecisionTreeClassifier(max_depth=-2)
        end
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
