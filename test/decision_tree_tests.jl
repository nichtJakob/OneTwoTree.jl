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
    leaf_842 = Node(prediction=842)
    leaf_2493 = Node(prediction=2493)
    leaf_683 = Node(prediction=683)

    decision_node_equals_161 = Node(
        decision = DecisionFn(x -> x == 161.0, "== 161.0"),  # x == 161.0
        true_child = leaf_2493,
        false_child = leaf_842
    )

    decision_node_less_than_28 = Node(
        decision = DecisionFn(x -> lessThan(x, 28.0), 28.0), # x < 28.0
        true_child = leaf_683,
        false_child = decision_node_equals_161
    )

    tree = DecisionTree(root=decision_node_less_than_28, max_depth=3)

    returned_string = tree_to_string(tree)

    expected_string = """
x < 28.0 ?
├─ False: x == 161.0 ?
│  ├─ False: 842.0
│  └─ True: 2493.0
└─ True: 683.0
"""

    @test returned_string == expected_string
end