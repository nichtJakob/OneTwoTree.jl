using OneTwoTree
using Test

@testset "DecisionTree Struct" begin
    t0 = DecisionTreeClassifier()
    @test t0.root === nothing
    @test t0.max_depth === -1

    t1 = DecisionTreeClassifier(max_depth=5)
    @test t1.root === nothing
    @test t1.max_depth === 5

    dataset = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # A 3x2 matrix of Real numbers
    labels = ["yes", "no", "yes"]
    n2 = Node(dataset, labels, true)

    t2 = DecisionTreeClassifier(root=n2, max_depth=5)
    @test t2.root === n2
    @test t2.max_depth === 5

    t3 = DecisionTreeClassifier(root=n2)
    @test t3.root === n2
    @test t3.max_depth === -1
end

function get_test_Tree_less_0_5() # returns 1.0 if the input in dim 1 is less 0.5 else 0.0
    # TODO: I changed the Node signature, so this needs to be updated as well
    # leaf1 = Node(prediction=1.0)
    # leaf2 = Node(prediction=0.0)
    # root = Node(decision = x -> lessThan(x, 0.5), true_child = leaf1, false_child = leaf2)
    return root
end


@testset "Tree.jl" begin # Tests the functionality of Node, tree_prediction, less in Tree.jl
    # @testset "Tree Prediction" begin
    #     root = get_test_Tree_less_0_5()
    #     @test tree_prediction(root, [1.0]) == 0.0
    #     @test tree_prediction(root, [0.0]) == 1.0
    #     @test tree_prediction(root, [55.0]) == 0.0
    #     @test tree_prediction(root, [-1.0]) == 1.0
    # end
end

# @testset "Print Tree" begin # Test: Tree with multiple decision nodes
#     leaf1 = Node(prediction=842)
#     leaf2 = Node(prediction=2493)
#     leaf3 = Node(prediction=683)

#     decision_node1 = Node(
#         decision = x -> x < 28,
#         decision_string = "x < 28",
#         true_child = leaf3,
#         false_child = leaf1
#     )

#     decision_node2 = Node(
#         decision = x -> x < 161,
#         decision_string = "x < 161",
#         true_child = leaf2,
#         false_child = decision_node1
#     )

#     tree = DecisionTree(root=decision_node2, max_depth=3)

#     #TODO:  Capture the printed output
#     #TODO: this does not work
#     #TODO: also if you use LLMs, you need to copy all your prompts into a txt file
#     output = capture_stdout() do
#         print_tree(tree)
#         @test tree_prediction(root, [-1.0]) == 1.0
#     end

#     # Expected output string with the exact structure
#     expected_output = """
# x < 161 ?
# ├─ False: x < 28 ?
# │   ├─ False: 842.0
# │   └─ True: 683.0
# └─ True: 2493.0
# """

#     # Check if the exact expected output matches the printed output
#     @test output == expected_output
# end