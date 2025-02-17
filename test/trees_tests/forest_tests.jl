### Tests for the creating forests.
### Test all kind of edge case input parameters, check the constructed forest variables
### after fitting, and check if the forest predictions make sense.

using Test
using OneTwoTree

@testset "Random Forests" begin
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

    @testset "Classifier" begin
        # Most basic "Doesn't crash" test
        #println("vor Forest erstellung")
        f0 = ForestClassifier(n_trees=1, n_features_per_tree=2, max_depth=1)
        #println("nach Forest erstellung")
        @test f0.n_trees == 1
        @test f0.n_features_per_tree == 2
        @test f0.max_depth == 1
        #println("vor fit!")
        fit!(f0, dataset1, cat_labels1)
        #println("nach fit!")
        #println("length(f0.trees) == 1 ? result: $(length(f0.trees))")
        @test length(f0.trees) == 1
        #println("f0.trees[1].max_depth == 1 ? result: $(f0.trees[1].max_depth))")
        @test f0.trees[1].max_depth == 1

        pred = predict(f0, dataset1)
        #println("length(pred) == length(cat_labels1) ? resultlenth(pred): $(length(pred)) result length(cat_labels1) == $(length(cat_labels1))")
        @test length(pred) == length(cat_labels1)
        #println("calc_accuracy(cat_labels1, pred) == 1.0 ? result: $(calc_accuracy(cat_labels1, pred)))")
        #@test calc_accuracy(cat_labels1, pred) == 1.0
        @test isapprox(calc_accuracy(cat_labels1, pred), 0.5, atol=0.5)

        #TODO: check n_features_per_tree out of bounds
        #TODO: check max_depth -1, -2, too large, too small etc.
        #TODO: check datatypes: Int, Float, String, Mixed
        #println("f0 Print: ---------------------------------------------------")
        #print_forest(f0)

        t_mixfs = DecisionTreeClassifier(max_depth=3)
        fit!(t_mixfs, dataset_mixfs, abcd_labels)
        pred_mixfs = predict(t_mixfs, dataset_mixfs)
        @test length(pred_mixfs) == 4
        @test calc_accuracy(abcd_labels, pred_mixfs) > 0.2
    end

    #@testset "Printing" begin
    #    fprint = ForestClassifier(n_trees=5, n_features_per_tree=6, max_depth=5)
    #    fit!(fprint, dataset1, cat_labels1)
    #    print_forest(fprint)
    #end


    @testset "Regression forest test" begin
        X_train = [1 2 3; 9 12 99; 400 20 -19]
        y_train = [-5.0, 20.5, 999.001]

        forest = ForestRegressor(n_trees=5, n_features_per_tree=40, max_depth=30)
        fit!(forest, X_train, y_train)

        X_test = [2.4; 3.0; -9.2]
        y_pred = predict(forest, X_test)

        #println(y_pred)
        @test isa(y_pred, Float64)
        @test y_pred <= 9999.001
        @test y_pred >= -5.0
    end

    @testset "Argument Errors" begin
        @test_throws ArgumentError ForestClassifier(n_trees=0, n_features_per_tree=1, max_depth=1)
        @test_throws ArgumentError ForestClassifier(n_trees=1, n_features_per_tree=0, max_depth=1)
        @test_throws ArgumentError ForestClassifier(n_trees=1, n_features_per_tree=1, max_depth=-2)
        ForestClassifier(n_trees=1, n_features_per_tree=1, max_depth=-1)
        ForestClassifier(n_trees=1, n_features_per_tree=1)

        forest = ForestRegressor(n_trees=5, n_features_per_tree=40, max_depth=30)
        X_test = [2.4; 3.0; -9.2]
        @test_throws ArgumentError predict(forest, X_test)
    end

    @testset "Printing forests" begin

        function tolerance(text::String)
            return replace(text, r"[\s\n]+" => "")
        end 

        X_train = reshape([42], 1, 1)
        y_train = [5.0]

        forest = ForestRegressor(n_trees=2, n_features_per_tree= 5, max_depth=3)
        fit!(forest, X_train, y_train)

        expected_output = tolerance("Tree 1:\n\nPrediction: 5.0\n\n\nTree 2:\n\nPrediction: 5.0\n\n")

        #test _forest_to_string(forest::AbstractForest)
        @test tolerance(tolerance(OneTwoTree._forest_to_string(forest))) == expected_output


        #test print_forest(forest::AbstractForest; io::IO=stdout)
        function get_print_forest(forest)
            buffer_1 = IOBuffer()
            print_forest(forest, io=buffer_1)
            output = String(take!(buffer_1))
        end

        printed_forest = get_print_forest(forest)
        @test tolerance(printed_forest) == expected_output


        #test Base.show(io::IO, forest::AbstractForest)
        function get_show_forest(forest)
            buffer_1 = IOBuffer()
            show(buffer_1, forest)
            output = String(take!(buffer_1))
        end

        shown_forest = get_show_forest(forest)
        @test tolerance(shown_forest) == expected_output
    end
    
end