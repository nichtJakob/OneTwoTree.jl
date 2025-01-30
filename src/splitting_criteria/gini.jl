"""
    gini_impurity(features::AbstractVector, labels::Vector{Bool}, decision_fn::Function) -> Float64

This function calculates the Gini impurity for a split in a decision tree.

# Arguments:
- `features`: A vector of features (e.g., true/false values or more complex data points).
- `labels`: A vector of Boolean labels indicating the target values (true/false).
- `decision_fn`: A function that takes a feature and returns `true` or `false` to define the split.
- `decision_param`: The parameter of the decision. This could be a numeric threshold e.g. 5 in x <= 5 or a class name "Ship" in x == "Ship".
- `decision_feature`: The index of the feature the decision function branches on. For datapoints with 5 features, this can be an index i in [1:5].

# Returns:
- The Gini impurity of the split.
"""

function gini_impurity(features::AbstractMatrix, labels::AbstractVector, node_data::Vector{Int64}, decision_fn::Function, decision_param::Union{Real, String}, decision_feature::Int64)::Float64

    # Filter features and labels using node_data
    # Split data in true and false
    split_true, split_false = split_indices(features, node_data, decision_fn, decision_param, decision_feature)

    # Labeling data
    true_labels = labels[split_true]
    false_labels = labels[split_false]

    # Handle empty labels edge case
    if isempty(labels) || (isempty(true_labels) && isempty(false_labels))
        return -1.0
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

    # Calculate proportions
    total_true = length(true_labels)
    total_false = length(false_labels)

    # Gini impurity for the true and false splits
    gini_true = 0.0
    gini_false = 0.0
    if total_true != 0
        gini_true = 1.0 - sum((count / total_true)^2 for count in values(label_counts_true))
    end
    if total_false != 0
        gini_false = 1.0 - sum((count / total_false)^2 for count in values(label_counts_false))
    end

    # Weighted Gini impurity
    total_length_data = length(node_data)
    gini_total = (total_true / total_length_data) * gini_true + (total_false / total_length_data) * gini_false

    return gini_total
end

function gini_impurity(features::AbstractMatrix, labels::AbstractVector, node_data::Vector{Int64})::Float64

    # Filter features and labels using node_data
    gini_labels = labels[node_data]
    total_labels = length(gini_labels)

    # Handle empty labels edge case
    if isempty(gini_labels)
        return 0
    end

    # Count occurrences of each label in true_labels and false_labels
    label_counts = Dict{Union{Real, String}, Int}()

    for label in gini_labels
        label_counts[label] = get(label_counts, label, 0) + 1
    end

    # Gini impurity for the dataset
    gini = 1.0 - sum((count / total_labels)^2 for count in values(label_counts))

    return gini
end

"""
    gini_gain(features::AbstractMatrix, labels::AbstractVector, node_data::Vector{Int64}, 
              decision_fn::Function, decision_param::Union{Real, String}, decision_feature::Int64) -> Float64

Computes the Gini gain of a potential split in a decision tree.

## Parameters
- `features::AbstractMatrix`: A matrix where each row represents a data sample and each column represents a feature.
- `labels::AbstractVector`: A vector of class labels corresponding to the rows in `features`.
- `node_data::Vector{Int64}`: A vector of indices representing the data points at the current node.
- `decision_fn::Function`: A function that determines the split criterion.
- `decision_param::Union{Real, String}`: A parameter for the `decision_fn`, such as a threshold value or category.
- `decision_feature::Int64`: The index of the feature to be used for the split.

## Returns
- `Float64`: The Gini gain, which quantifies the reduction in impurity when the node is split using the given decision criterion.

## Description
The Gini gain is calculated as the difference between the Gini impurity of the current node and the weighted Gini impurity of the child nodes after applying the split.

## Example
```julia
gain = gini_gain(features, labels, node_data, (x, t) -> x < t, 0.5, 2)

"""

function gini_gain(features::AbstractMatrix, labels::AbstractVector, node_data::Vector{Int64}, decision_fn::Function, decision_param::Union{Real, String}, decision_feature::Int64)::Float64
    return gini_impurity(features, labels, node_data) - gini_impurity(features, labels, node_data, decision_fn, decision_param, decision_feature)
end

# new signature
function gini_gain(parent_labels::AbstractVector, true_child_labels::AbstractVector, false_child_labels::AbstractVector)::Float64
    return gini_impurity(parent_labels) - gini_impurity(true_child_labels, false_child_labels)
end

function gini_impurity(labels::AbstractVector)::Float64
    total_labels = length(labels)

    # Handle empty labels edge case
    if isempty(labels)
        return 0
    end

    # Count occurrences of each label in true_labels and false_labels
    label_counts = Dict{Union{Real, String}, Int}()

    for label in labels
        label_counts[label] = get(label_counts, label, 0) + 1
    end

    # Gini impurity for the dataset
    gini = 1.0 - sum((count / total_labels)^2 for count in values(label_counts))

    return gini
end

function gini_impurity(true_child_labels::AbstractVector, false_child_labels::AbstractVector)::Float64

    # Handle empty labels edge case
    if (isempty(true_child_labels) && isempty(false_child_labels))
        return -1.0
    end

    # Count occurrences of each label in true_labels and false_labels
    label_counts_true = Dict{Union{Real, String}, Int}()
    label_counts_false = Dict{Union{Real, String}, Int}()

    for label in true_child_labels
        label_counts_true[label] = get(label_counts_true, label, 0) + 1
    end

    for label in false_child_labels
        label_counts_false[label] = get(label_counts_false, label, 0) + 1
    end

    # Calculate proportions
    total_true = length(true_child_labels)
    total_false = length(false_child_labels)
    total = total_true + total_false

    # Gini impurity for the true and false splits
    gini_true = 0.0
    gini_false = 0.0
    if total_true != 0
        gini_true = 1.0 - sum((count / total_true)^2 for count in values(label_counts_true))
    end
    if total_false != 0
        gini_false = 1.0 - sum((count / total_false)^2 for count in values(label_counts_false))
    end

    # Weighted Gini impurity
    gini_total = (total_true / total) * gini_true + (total_false / total) * gini_false

    return gini_total
end