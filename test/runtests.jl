using OneTwoTree
using Test

function get_test_Tree_less_0_5() # returns 1.0 if the input in dim 1 is less 0.5 else 0.0
    # TODO: I changed the Node signature, so this needs to be updated as well
    # leaf1 = Node(prediction=1.0)
    # leaf2 = Node(prediction=0.0)
    # root = Node(decision = x -> lessThan(x, 0.5), true_child = leaf1, false_child = leaf2)
    return root
end


@testset "Tree.jl" begin # Tests the functionality of Node, tree_prediction, less in Tree.jl

    @testset "tree_creation" begin
        dataset = [[3,4] [6,1] [0,2]]
        cat_labels = ["Chicken", "Egg"]
        reg_labels = [12.0, 4.5]

        root = Node(dataset, cat_labels, true)
        root = Node(dataset, reg_labels, false)

        dataset1 = [[3.5,1.0,5.6] [9.1,1.2,3.3] [2.9,0.4,4.3]]
        dataset2 = [["Snow","Lax","Arm"] ["Hard","Snow","Hard"] ["Arm","Page","Payoff"]]
        dataset3 = [[3.1,"Lax","Arm"] [0.6,"Snow","Hard"] [4.2,"Page","Payoff"]]
        cat_labels1 = ["Chicken", "Egg", "Egg"]
        reg_labels1 = [12.0, 4.5, 6.7]

        root = Node(dataset1, cat_labels1, true)
        root = Node(dataset1, reg_labels1, false)
        root = Node(dataset1, cat_labels1, true, max_depth=1)
        root = Node(dataset1, reg_labels1, false, max_depth=1)
        root = Node(dataset2, cat_labels1, true, max_depth=1)
        root = Node(dataset2, reg_labels1, false, max_depth=1)
        # @test tree_prediction(root, [1.0]) == 0.0
        # @test tree_prediction(root, [0.0]) == 1.0
        # @test tree_prediction(root, [55.0]) == 0.0
        # @test tree_prediction(root, [-1.0]) == 1.0
    end

    @testset "tree_prediction" begin
        # TODO: I changed the Node signature, so this needs to be updated as well
        # root = get_test_Tree_less_0_5()
        # @test tree_prediction(root, [1.0]) == 0.0
        # @test tree_prediction(root, [0.0]) == 1.0
        # @test tree_prediction(root, [55.0]) == 0.0
        # @test tree_prediction(root, [-1.0]) == 1.0
    end
end
