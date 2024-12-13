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

        @test tree_prediction(root, [-1.0]) == 1.0   
    end 
end


@testset "Test 1: Exact Output Matching" begin # Test: Tree with multiple decision nodes
    leaf1 = Node(prediction=842)
    leaf2 = Node(prediction=2493)
    leaf3 = Node(prediction=683)

    decision_node1 = Node(
        decision = x -> x < 28,
        decision_string = "x < 28",
        true_child = leaf3,
        false_child = leaf1
    )

    decision_node2 = Node(
        decision = x -> x < 161,
        decision_string = "x < 161",
        true_child = leaf2,
        false_child = decision_node1
    )

    tree = DecisionTree(decision_node2, 3)

    # Capture the printed output
    output = capture_stdout() do
        print_tree(tree)
    end

    # Expected output string with the exact structure
    expected_output = """
x < 161 ?
├─ False: x < 28 ?
│   ├─ False: 842.0
│   └─ True: 683.0
└─ True: 2493.0
"""

    # Check if the exact expected output matches the printed output
    @test output == expected_output
end
