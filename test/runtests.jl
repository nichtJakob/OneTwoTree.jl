using OneTwoTree
using Test

function get_test_Tree_less_0_5() # returns 1.0 if the input in dim 1 is less 0.5 else 0.0
    leaf1 = Node(prediction=0.0)
    leaf2 = Node(prediction=1.0)
    root = Node(decision_is_right = x -> lessThan(x, 0.5), left_child = leaf1, right_child = leaf2)
    
    return root
end


@testset "OneTwoTree.jl" begin

    @testset "Node" begin
        root = get_test_Tree_less_0_5()
        @test tree_prediction(root, [1.0]) == 0.0 
        @test tree_prediction(root, [0.0]) == 1.0
        @test tree_prediction(root, [55.0]) == 0.0
        @test tree_prediction(root, [-1.0]) == 1.0   
    end 



end
