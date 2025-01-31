### Test DecisionTree constructors and print function

using OneTwoTree
using Test

@testset "DecisionTree Struct" begin
    t0 = DecisionTreeClassifier()
    @test isnothing(t0.root)
    @test t0.max_depth === -1

    t1 = DecisionTreeClassifier(max_depth=5)
    @test isnothing(t1.root)
    @test t1.max_depth === 5

    dataset = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    labels = ["yes", "no", "yes"]
    n2 = OneTwoTree.Node(dataset, labels, true)

    t2 = DecisionTreeClassifier(root=n2, max_depth=5)
    @test t2.root === n2
    @test t2.max_depth === 5

    t3 = DecisionTreeClassifier(root=n2)
    @test t3.root === n2
    @test t3.max_depth === -1
end

@testset "Print Tree" begin # Test: stringify tree with multiple decision nodes

    @testset "Basic" begin
        dataset = reshape([
            1.0;
            9.0
        ], 2, 1)
        labels = ["A", "B"]

        t = DecisionTreeClassifier(max_depth=1)
        fit!(t, dataset, labels)

        returned_string = OneTwoTree._tree_to_string(t, false)
        expected_string = "
x[1] <= 5.0 ?
├─ True:  A
└─ False: B
"
        @test returned_string == expected_string
    end

    @testset "Depth 2" begin
        dataset1 = reshape([
            1.0;
            3.0;
            5.0
        ], 3, 1)
        labels1 = ["A", "B", "C"]

        t = DecisionTreeClassifier(max_depth=2)
        fit!(t, dataset1, labels1)

        returned_string = OneTwoTree._tree_to_string(t, false)
        expected_string = "
x[1] <= 2.0 ?
├─ True:  A
└─ False: x[1] <= 4.0 ?
   ├─ True:  B
   └─ False: C
"

        @test returned_string == expected_string
    end

    @testset "Depth 3" begin
        dataset2 = [
            1.0 2.0 3.0
            1.0 2.0 4.0
            1.0 -2.0 3.0
            1.0 -2.0 4.0
            -1.0 2.0 3.0
        ]
        labels2 = ["A", "B", "C", "D", "E"]

        t = DecisionTreeClassifier()
        fit!(t, dataset2, labels2)

        returned_string = OneTwoTree._tree_to_string(t, false)
        expected_string = "
x[1] <= 0.0 ?
├─ True:  E
└─ False: x[2] <= 0.0 ?
   ├─ True:  x[3] <= 3.5 ?
   │  ├─ True:  C
   │  └─ False: D
   └─ False: x[3] <= 3.5 ?
      ├─ True:  A
      └─ False: B
"
        # println(returned_string)
        # println(cmp(expected_string, returned_string))
        @test returned_string == expected_string
    end


    @testset "print_tree" begin
        function tolerance(text::String)
            return replace(text, r"[\s\n]+" => "")
        end 

        X_train = reshape([12], 1, 1)
        y_train = ["A"]

        tree = DecisionTreeClassifier()
        fit!(tree, X_train, y_train)

        expected_output = tolerance("Tree(max_depth=-1)Prediction:A")
        expected_output_short = tolerance("Prediction:A")
        
        #test _tree_to_string()
        @test tolerance(OneTwoTree._tree_to_string(tree)) == expected_output


        #test print_tree()
        function get_print_tree(ftree)
            buffer_1 = IOBuffer()
            print_tree(tree, io=buffer_1)
            output = String(take!(buffer_1))
        end

        printed_tree = get_print_tree(tree)
        @test tolerance(printed_tree) == expected_output_short
        

        #test Base.show()
        function get_show_tree(tree)
            buffer_1 = IOBuffer()
            show(buffer_1, tree)
            output = String(take!(buffer_1))
        end

        shown_tree = get_show_tree(tree)
        @test tolerance(shown_tree) == expected_output
    end

end

@testset "Tree Argument Errors" begin
    @test_throws ArgumentError OneTwoTree.DecisionTreeRegressor(max_depth= -5)
    @test_throws ArgumentError OneTwoTree.DecisionTreeClassifier(max_depth= -5)

    # test Argument errors on function _verify_fit!_args(tree, dataset, labels, column_data)
    tree = DecisionTreeClassifier()
    tree_regressor = DecisionTreeRegressor()
    dataset = [1.0 2.0; 3.0 4.0;]
    labels = ["yes", "no"]
    column_data = false

    @test_throws ArgumentError OneTwoTree._verify_fit!_args(tree, dataset, [], column_data)
    @test_throws ArgumentError OneTwoTree._verify_fit!_args(tree, [], labels, column_data)
    #maxDepth @test_throws ArgumentError OneTwoTree._verify_fit!_args(tree, dataset, labels, column_data)
    @test_throws ArgumentError OneTwoTree._verify_fit!_args(tree, dataset, ["yes", "no", "yes"], false)
    @test_throws ArgumentError OneTwoTree._verify_fit!_args(tree, dataset, ["yes", "no", "yes"], true)
    @test_throws ArgumentError OneTwoTree._verify_fit!_args(tree, dataset, ["yes", 4], column_data)
    @test_throws ArgumentError OneTwoTree._verify_fit!_args(tree_regressor, dataset, labels, column_data)


    # test Argument errors on function predict(tree::AbstractDecisionTree, X::Union{AbstractMatrix, AbstractVector})
    @test_throws ArgumentError OneTwoTree.predict(tree, [15])


     # tests on calc_accuracy(labels::AbstractArray{S}, predictions::AbstractArray{T})
     @test_throws ArgumentError OneTwoTree.calc_accuracy([1, 2, 3, 4], [1, 2, 3, 4, 5])
     @test OneTwoTree.calc_accuracy([], []) == 0.0


     # tests on function calc_depth(tree::AbstractDecisionTree)
     @test OneTwoTree.calc_depth( OneTwoTree.DecisionTreeRegressor()) == 0
end
