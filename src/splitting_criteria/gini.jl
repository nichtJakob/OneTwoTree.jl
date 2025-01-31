"""
    gini_gain(parent_labels::AbstractVector, true_child_labels::AbstractVector, false_child_labels) -> Float64

This function calculates the gain in Gini impurity for a split in a decision tree. The split is characterized by the partition of the parent_labels into true_child_labels and false_child_labels according to some discriminant function.

# Arguments:
- `parent_labels`: A vector of data labels (e.g., classes or numerical values in the case of regression).
- `true_child_labels`: A vector of a subset of data labels contained in parent_labels.
- `false_child_labels`: A vector of a subset of data labels contained in parent_labels.

# Returns:
- The gain in Gini impurity of the split.
"""
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