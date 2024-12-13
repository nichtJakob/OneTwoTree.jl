using OneTwoTree
using Test

@testset "DecisionTree struct" begin
    t0 = DecisionTree()
    @test t0.root === nothing
    @test t0.max_depth === -1

    t1 = DecisionTree(max_depth=5)
    @test t1.root === nothing
    @test t1.max_depth === 5

    n2 = Node(prediction=1.0)
    t2 = DecisionTree(root=n2, max_depth=5)
    @test t2.root === n2
    @test t2.max_depth === 5

    t3 = DecisionTree(root=n2)
    @test t3.root === n2
    @test t3.max_depth === -1
end

@testset "Tree to string" begin # Test: stringify tree with multiple decision nodes
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

    tree = DecisionTree(root=decision_node2, max_depth=3)

    returned_string = tree_to_string(tree)

    expected_string = """
x < 161 ?
├─ False: x < 28 ?
│  ├─ False: 842.0
│  └─ True: 683.0
└─ True: 2493.0
"""

    @test returned_string == expected_string
end