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