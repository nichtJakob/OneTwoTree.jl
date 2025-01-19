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
        f0 = ForestClassifier(n_trees=1, n_features_per_tree=2, max_depth=1)
        @test f0.n_trees == 1
        @test f0.n_features_per_tree == 2
        @test f0.max_depth == 1

        fit!(f0, dataset1, cat_labels1)
        @test length(f0.trees) == 1
        @test f0.trees[1].max_depth == 1

        pred = predict(f0, dataset1)
        @test length(pred) == length(cat_labels1)
        @test calc_accuracy(cat_labels1, pred) == 1.0

        #TODO: check n_features_per_tree out of bounds
        #TODO: check max_depth -1, -2, too large, too small etc.
        #TODO: check datatypes: Int, Float, String, Mixed
    end

    @testset "Regressor" begin
        #TODO:
    end
end