"""
    gini_impurity(features::AbstractVector, labels::Vector{Bool}, decision_fn::Function) -> Float64

This function calculates the Gini impurity for a split in a decision tree.

# Arguments:
- `features`: A vector of features (e.g., true/false values or more complex data points).
- `labels`: A vector of Boolean labels indicating the target values (true/false).
- `decision_fn`: A function that takes a feature and returns `true` or `false` to define the split.

# Returns:
- The Gini impurity of the split.
"""

function gini_impurity(features::AbstractVector, labels::Vector{Bool}, decision_fn::Function)::Float64

    #Split data in true and false
    split_true = [i for i in eachindex(features) if decision_fn(features[i])]
    split_false = [i for i in eachindex(features) if !decision_fn(features[i])]

    #Labeling data
    true_labels = labels[split_true]
    false_labels = labels[split_false]

    
    # Handle empty labels edge case
    if isempty(labels) || (isempty(true_labels) && isempty(false_labels))
        
        return 0
    end

    #Calculate Gini


    #Number of true in true_labels and false_labels
    true_num_true = count(x -> x == true, true_labels)
    false_num_true = count(x -> x == true, false_labels)

    #Number of false in true_labels and false_labels
    true_num_false = count(x -> x == false, true_labels)
    false_num_false = count(x -> x == false, false_labels)

    #Calculate proportions
    total_true = length(true_labels)
    total_false = length(false_labels)

    #Gini for true nod
    gini_true = 1 - (true_num_true/total_true)^2 - (true_num_false/total_true)^2

    #Gini for false nod
    gini_false = 1 - (false_num_true/total_false)^2 - (false_num_false/total_false)^2

    #weighted gini
    total_length_data = length(features)
    gini_total = length(split_true)/total_length_data * gini_true + length(split_false)/total_length_data * gini_false

    return gini_total


end 






using Test

function test_gini_impurity()
    println("Running gini_impurity tests...")

    # Test 1: Basic binary features
    features1 = [true, false, true, true, false]
    labels1 = [true, false, true, false, false]
    decision_fn1 = x -> x == true
    gini1 = gini_impurity(features1, labels1, decision_fn1)
    @test isapprox(gini1, 0.266, atol=1e-2)

    # Test 2: Numerical features
    features2 = [25, 40, 35, 22, 60]
    labels2 = [true, false, true, false, true]
    decision_fn2 = x -> x > 30
    gini2 = gini_impurity(features2, labels2, decision_fn2)
    @test isapprox(gini2, 0.466, atol=1e-2)

    # Test 3: Empty features and labels
    features3 = Int[]
    labels3 = Bool[]
    decision_fn3 = x -> x > 30
    gini3 = gini_impurity(features3, labels3, decision_fn3)
    @test gini3 == 0.0

    # Test 4: All labels are the same
    features4 = [1, 2, 3, 4, 5]
    labels4 = [true, true, true, true, true]
    decision_fn4 = x -> x > 3
    gini4 = gini_impurity(features4, labels4, decision_fn4)
    @test gini4 == 0.0

    # Test 5: Perfect split
    features5 = [1, 2, 3, 4, 5, 6]
    labels5 = [true, true, true, false, false, false]
    decision_fn5 = x -> x <= 3
    gini5 = gini_impurity(features5, labels5, decision_fn5)
    @test gini5 == 0.0

    # Test 6: Uneven split with imbalance
    features6 = [10, 20, 30, 40, 50]
    labels6 = [true, true, false, false, false]
    decision_fn6 = x -> x < 35
    gini6 = gini_impurity(features6, labels6, decision_fn6)
    @test isapprox(gini6, 0.266, atol=1e-2)

    println("All tests passed!")
end

test_gini_impurity()
