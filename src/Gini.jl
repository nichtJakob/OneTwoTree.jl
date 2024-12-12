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

function gini_impurity(features::Vector{Union{Real, String}}, labels::Vector{Union{Real, String}}, node_data::Vector{Int64}, decision_fn::Function)::Float64
#function gini_impurity(features::Vector{Bool}, labels::Vector{Bool}, node_data::Vector{Int64}, decision_fn::Function)::Float64 
# Filter features and labels using node_data
    features = features[node_data]
    labels = labels[node_data]
    
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

    # Handle empty labels edge case
    if isempty(true_labels) || isempty(false_labels)
        return 0.0  # Return 0 if one of the splits is empty
    end


    #Number of true in true_labels and false_labels
    true_num_true = count(x -> x == decision_fn(x), true_labels)
    false_num_true = count(x -> x == decision_fn(x), false_labels)

    #Number of false in true_labels and false_labels
    true_num_false = count(x -> x == decision_fn(x), true_labels)
    false_num_false = count(x -> x == decision_fn(x), false_labels)

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

@testset "Gini Impurity Tests with Non-Boolean Labels" begin
    # Test 1: Integer labels
    @testset "Test with Integer Labels" begin
        features1 = Union{Real, String}[1, 2, 3, 4]
        labels1 = Union{Real, String}[1, 0, 1, 0]  # Integer labels
        node_data1 = [1, 2, 3, 4]
        decision_fn1 = x -> x <= 2
        gini1 = gini_impurity(features1, labels1, node_data1, decision_fn1)
        @test isapprox(gini1, 0.5, atol=1e-2)  # Replace with expected result
    end

    # Test 2: String labels
    @testset "Test with String Labels" begin
        features2 = Union{Real, String}["high", "low", "medium", "high"]
        labels2 = Union{Real, String}["yes", "no", "yes", "no"]  # String labels
        node_data2 = [1, 2, 3, 4]
        decision_fn2 = x -> x == "high"
        gini2 = gini_impurity(features2, labels2, node_data2, decision_fn2)
        print(gini2)
        @test isapprox(gini2, 0.5, atol=1e-2)  # Replace with expected result
    end
end
