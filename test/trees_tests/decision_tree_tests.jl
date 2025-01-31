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

    function get_print_tree(tree)
        buffer_1 = IOBuffer()
        print_tree(tree, io=buffer_1)
        output = String(take!(buffer_1))
    end

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

    printed_tree = get_print_tree(t)
    @test printed_tree != ""
end