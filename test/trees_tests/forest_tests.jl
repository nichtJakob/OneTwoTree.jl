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
    end

    #@testset "Printing" begin
    #    fprint = ForestClassifier(n_trees=5, n_features_per_tree=6, max_depth=5)
    #    fit!(fprint, dataset1, cat_labels1)
    #    print_forest(fprint)
    #end


    @testset "Regressor" begin
        #TODO:
    end
end