
function variance(data::AbstractVector)::Float64
    if isempty(data)
        return 0.0
    end
    mean_value = mean(data)
    variance = mean((data .- mean_value) .^ 2)
    return variance
end

"""
    variance_gain(parent_labels::AbstractVector, true_child_labels::AbstractVector, false_child_labels::AbstractVector)::Float64

This function calculates the variance gain for a split in a decision tree. The split is characterized by the partition of the parent_labels into true_child_labels and false_child_labels according to some discriminant function.

# Arguments:
- `parent_labels`: A vector of data labels (e.g., classes or numerical values in the case of regression).
- `true_child_labels`: A vector of a subset of data labels contained in parent_labels.
- `false_child_labels`: A vector of a subset of data labels contained in parent_labels.

# Returns:
- The variance gain of the split.
"""
function variance_gain(parent_labels::AbstractVector, true_child_labels::AbstractVector, false_child_labels::AbstractVector)::Float64
    max_gain = variance(parent_labels)

    if isempty(true_child_labels) || isempty(false_child_labels) || max_gain == 0.0
        return 0.0
    end

    total = length(parent_labels)

    true_child_weight = length(true_child_labels) / total
    false_child_weight = length(false_child_labels) / total

    true_weighted_var = true_child_weight * variance(true_child_labels)
    false_weighted_var = false_child_weight * variance(false_child_labels)
    weighted_var = true_weighted_var + false_weighted_var

    gain = max_gain - weighted_var

    # normalization not necessary but cool
    normalized_gain = gain / max_gain

    return normalized_gain
end