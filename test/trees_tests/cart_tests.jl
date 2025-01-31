### Tests for the CART algorithm.
### The CART algorithm constructs a DecisionTree from given data.
### Test use cases and different data types.

using Test
using OneTwoTree
using Suppressor # suppress prints in tests


const RUN_MNIST = false
const USE_INT_FEATURES = true

"""
    test_node_consistency(node::Node)

Checks if the properties of the node are consistent with the type of the node.
"""
function test_node_consistency(node::OneTwoTree.Node)
    if OneTwoTree.is_leaf(node)
        @test node.prediction isa Number || node.prediction isa String
        @test isnothing(node.true_child)
        @test isnothing(node.false_child)
        @test isnothing(node.decision)
    else
        @test isnothing(node.prediction)
        @test node.true_child isa OneTwoTree.Node
        @test node.false_child isa OneTwoTree.Node
        @test node.decision isa OneTwoTree.Decision
    end
end

"""
    test_no_duplicates(node::OneTwoTree.Node)
Tests that no two node children classify the same (makes no sense)
"""
function test_no_duplicates(node::OneTwoTree.Node)
    if OneTwoTree.is_leaf(node)
        return
    end

    if OneTwoTree.is_leaf(node.true_child) && OneTwoTree.is_leaf(node.false_child)
        @test node.true_child.prediction != node.false_child.prediction
    end
end


"""
    test_tree_consistency(tree, run_tests::Bool=true)

Traverses the tree and checks all properties of the tree and its nodes for consistency.
"""
function test_tree_consistency(; tree::OneTwoTree.AbstractDecisionTree, run_tests::Bool=true)
    if !run_tests
        @warn "Skipping tree consistency tests"
        return
    end

    # test tree properties
    @test tree.max_depth > 0 || tree.max_depth == -1

    if isnothing(tree.root)
        return
    end

    # test node integrity
    to_visit = [tree.root]
    while !isempty(to_visit)
        node = popfirst!(to_visit)

        test_node_consistency(node)
        test_no_duplicates(node)

        if !isnothing(node.true_child)
            push!(to_visit, node.true_child)
        end

        if !isnothing(node.false_child)
            push!(to_visit, node.false_child)
        end
    end

    # test depth consistency
    if tree.max_depth > 0
        @test OneTwoTree.calc_depth(tree) <= tree.max_depth
    end
end

"""
Calls CART on simple manually crafted datasets for classification trees.
Tests consistency of all data fields in constructed trees.
Tests different and also mixed data types.
Tests whether constructed trees predict expected values and conform to given constraints
(e.g. max_depth).
"""

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
    dataset_mixfs = [
        7 "Old" 4 "Rich"
        3 "Young" 4 "Poor"
        3 "Young" 3 "Middle-class"
        1 "Middle-aged" 7 "Middle-class"
    ]
    abc_labels = ["A", "B", "C"]
    abcd_labels = ["A", "B", "C", "D"]
    aabcbb_labels = ["A", "A", "B", "C", "B", "B"]
    @testset "Fit and Predict" begin
        t1 = DecisionTreeClassifier(max_depth=1)
        fit!(t1, dataset1, cat_labels1)

        @test t1.root isa OneTwoTree.Node
        @test t1.max_depth == 1
        test_tree_consistency(tree=t1, run_tests=!isnothing(t1.root))

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

    @testset "Data Types" begin
        if !USE_INT_FEATURES
            dataset_int = convert(Matrix{Float64}, dataset_int)
        end

        t_float = DecisionTreeClassifier(max_depth=3)
        t_string = DecisionTreeClassifier(max_depth=3)
        t_int = DecisionTreeClassifier(max_depth=3)
        t_mixfs = DecisionTreeClassifier(max_depth=3)

        fit!(t_float, dataset_float, abc_labels)
        fit!(t_string, dataset_string,  abc_labels)
        fit!(t_int, dataset_int, aabcbb_labels)
        fit!(t_mixfs, dataset_mixfs, abcd_labels)

        @test t_float.root isa OneTwoTree.Node
        @test t_string.root isa OneTwoTree.Node
        @test t_int.root isa OneTwoTree.Node
        @test t_mixfs.root isa OneTwoTree.Node
        test_tree_consistency(tree=t_float, run_tests=!isnothing(t_float.root))
        test_tree_consistency(tree=t_string, run_tests=!isnothing(t_string.root))
        test_tree_consistency(tree=t_int, run_tests=!isnothing(t_int.root))
        test_tree_consistency(tree=t_mixfs, run_tests=!isnothing(t_mixfs.root))
        @test OneTwoTree.calc_depth(t_float) == 2
        @test OneTwoTree.calc_depth(t_string) == 2
        @test OneTwoTree.calc_depth(t_int) == 2
        @test OneTwoTree.calc_depth(t_mixfs) == 3

        pred_float = predict(t_float, dataset_float)
        pred_string = predict(t_string, dataset_string)
        pred_int = predict(t_int, dataset_int)
        pred_mixfs = predict(t_mixfs, dataset_mixfs)

        @test length(pred_float) == 3
        @test length(pred_string) == 3
        @test length(pred_int) == 6
        @test length(pred_mixfs) == 4
        @test calc_accuracy(abc_labels, pred_float) == 1.0
        @test calc_accuracy(abc_labels, pred_string) == 1.0
        @test calc_accuracy(aabcbb_labels, pred_int) == 1.0
        @test calc_accuracy(abcd_labels, pred_mixfs) == 1.0
    end

    @testset "Int Label" begin
        t_int_label = DecisionTreeClassifier(max_depth=3)
        fit!(t_int_label, dataset_float, [1, 2, 3])

        @test t_int_label.root isa OneTwoTree.Node
        test_tree_consistency(tree=t_int_label, run_tests=!isnothing(t_int_label.root))
        @test OneTwoTree.calc_depth(t_int_label) == 2

        pred_int_label = predict(t_int_label, dataset_float)
        @test length(pred_int_label) == 3
        @test calc_accuracy([1, 2, 3], pred_int_label) == 1.0
    end

    @testset "Max Depth" begin
        @testset "Zero Depth" begin
            t_zero_depth = DecisionTreeClassifier(max_depth=0)
            fit!(t_zero_depth, dataset_float, abc_labels)
            @test OneTwoTree.calc_depth(t_zero_depth) == 0
        end

        # unlimited depth
        @testset "Unlimited Depth" begin
            t_unlimited = DecisionTreeClassifier(max_depth=-1)
            fit!(t_unlimited, dataset_float, abc_labels)

            @test t_unlimited.root isa OneTwoTree.Node
            test_tree_consistency(tree=t_unlimited, run_tests=!isnothing(t_unlimited.root))
            @test OneTwoTree.calc_depth(t_unlimited) > 1

            pred_unlimited = predict(t_unlimited, dataset_float)
            @test length(pred_unlimited) == length(abc_labels)
            @test calc_accuracy(abc_labels, pred_unlimited) == 1.0
        end

        @testset "Bigger Depth Than Necessary" begin
            t_bigger = DecisionTreeClassifier(max_depth=12)
            fit!(t_bigger, dataset_float, abc_labels)

            @test t_bigger.root isa OneTwoTree.Node
            test_tree_consistency(tree=t_bigger, run_tests=!isnothing(t_bigger.root))
            @test OneTwoTree.calc_depth(t_bigger) <= 2

            pred_bigger = predict(t_bigger, dataset_float)
            @test length(pred_bigger) == length(abc_labels)
            @test calc_accuracy(abc_labels, pred_bigger) == 1.0
        end

        @testset "Smaller Depth Than Necessary" begin
            t_smaller = DecisionTreeClassifier(max_depth=1)
            fit!(t_smaller, dataset_float, abc_labels)

            @test t_smaller.root isa OneTwoTree.Node
            test_tree_consistency(tree=t_smaller, run_tests=!isnothing(t_smaller.root))
            @test OneTwoTree.calc_depth(t_smaller) == 1

            pred_smaller = predict(t_smaller, dataset_float)
            @test length(pred_smaller) == length(abc_labels)
            @test calc_accuracy(abc_labels, pred_smaller) > 0.5
        end

        @testset "Invalid Depth" begin
            @test_throws "DecisionTreeClassifier: Got invalid max_depth. Set it to a value >= -1. (-1 means unlimited depth)" DecisionTreeClassifier(max_depth=-2)
        end
    end
end

@testset "Custom Splitting Criteria" begin
    dataset_zork = [
        500.0 57.3 399.0 -250.0
        34.2 21.3 781.5 994.0
        -402.0 63.0 333.3 443.1
        141.0 687.1 231.0 55.0
        125.0 462.2 425.6 154.0
        -392.0 366.2 -30.0 -220.0
        -94.0 72.0 482.0 6.0
        322.0 33.2 -230.0 750.0
    ]
    zork_labels = ["BAYALA", "BLORB", "CLEESH", "BAYALA", "LESOCH", "MUSDEX", "ZIMBOR", "MUSDEX"]

    @testset "Information Gain" begin
        tz = DecisionTreeClassifier(max_depth=3)
        fit!(tz, dataset_zork, zork_labels, splitting_criterion=information_gain)

        @test tz.root isa OneTwoTree.Node
        @test tz.max_depth == 3
        test_tree_consistency(tree=tz, run_tests=!isnothing(tz.root))

        pred = predict(tz, dataset_zork)
        @test length(pred) == length(zork_labels)
        @test calc_accuracy(zork_labels, pred) == 1.0
    end
end

"""
Runs the examples from the ReadMe to make sure that they work.
"""

@testset "ReadMe Examples" begin
    @testset "Classification" begin
        dataset = [
            3.5 9.1 2.9
            1.0 1.2 0.4
            5.6 3.3 4.3
        ]
        labels = ["A", "B", "C"]

        tree = DecisionTreeClassifier(max_depth=2)
        @capture_out print(tree)
        @capture_out print(fit!(tree, dataset, labels))
        @capture_out print(tree)
        prediction = predict(tree, [
            2.0 4.0 6.0
        ])
        @capture_out print("The tree predicted class $(prediction[1]).")
        @test true # no errors thrown
    end

    @testset "Regression" begin
        dataset = [
            1.0 2.0
            2.0 3.0
            3.0 4.0
            4.0 5.0
        ]
        labels = [1.5, 2.5, 3.5, 4.5]

        tree = DecisionTreeRegressor(max_depth=3)
        @capture_out print(tree)
        @capture_out print(fit!(tree, dataset, labels))
        @capture_out print(tree)

        prediction = predict(tree, [
            1.0 4.0
        ])
        @capture_out print("The tree predicted $(prediction[1]).")
        @test true # no errors thrown
    end
end
