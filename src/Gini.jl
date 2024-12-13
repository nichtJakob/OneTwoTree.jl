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

#function gini_impurity(features::Vector{Union{Real, String}}, labels::Vector{Union{Real, String}}, node_data::Vector{Int64}, decision_fn::Function)::Float64
function gini_impurity(features::AbstractVector, labels::AbstractVector, node_data::Vector{Int64}, decision_fn::Function)::Float64

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


    # Count occurrences of each label in true_labels and false_labels
    label_counts_true = Dict{Union{Real, String}, Int}()
    label_counts_false = Dict{Union{Real, String}, Int}()

    for label in true_labels
        label_counts_true[label] = get(label_counts_true, label, 0) + 1
    end

    for label in false_labels
        label_counts_false[label] = get(label_counts_false, label, 0) + 1
    end

    #Calculate proportions
    total_true = length(true_labels)
    total_false = length(false_labels)


    # Gini impurity for the true split
    gini_true = 1.0 - sum((count / total_true)^2 for count in values(label_counts_true))

    # Gini impurity for the false split
    gini_false = 1.0 - sum((count / total_false)^2 for count in values(label_counts_false))

    # Weighted Gini impurity
    total_length_data = length(features)
    gini_total = (length(split_true) / total_length_data) * gini_true + (length(split_false) / total_length_data) * gini_false


    return gini_total


end  
