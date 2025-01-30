#using StatsBase

function variance(data::AbstractVector) :: Float64
    #TODO different datatypes?
    if isempty(data)
        return 0.0
    end
    mean_value = mean(data)
    variance = mean((data .- mean_value) .^ 2)
    return variance
end

function variance(labels::AbstractVector, node_data::Vector{Int64}) :: Float64
    if isempty(labels)
        return 0.0
    end
    # TODO: here we copy a lot, but I wasnt too confident to change it on release day
    mean_value = mean(labels[node_data])
    variance = mean((labels[node_data] .- mean_value) .^ 2)
    return variance
end

# function variance(child_1_labels::AbstractVector, child_2_labels::AbstractVector) :: Float64
function variance(features::AbstractMatrix, labels::AbstractVector, node_data::Vector{Int64}, decision_fn::Function, decision_param::Union{Real, String}, decision_feature::Int64)::Float64

    # Split data in true and false
    split_true, split_false = split_indices(features, node_data, decision_fn, decision_param, decision_feature)

    # Labeling data
    true_labels = labels[split_true]
    false_labels = labels[split_false]
    total_labels = length(labels[node_data])

    if isempty(true_labels) || isempty(false_labels)
        return 0.0
    end

    true_weight = length(true_labels) / total_labels
    false_weight = length(false_labels) / total_labels

    weighted_var_1 = true_weight * variance(true_labels)
    weighted_var_2 = false_weight * variance(false_labels)
    weighted_var = weighted_var_1 + weighted_var_2

    return weighted_var
end

function variance_gain(features::AbstractMatrix, labels::AbstractVector, node_data::Vector{Int64}, decision_fn::Function, decision_param::Union{Real, String}, decision_feature::Int64)::Float64
    return variance(labels, node_data) - variance(features, labels, node_data, decision_fn, decision_param, decision_feature)
end

# # Splitting criterion for regression based on the variance similar to infogain
function variance_gain(parent_labels::AbstractVector, true_child_labels::AbstractVector, false_child_labels::AbstractVector) :: Float64
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

    # normalization not neccessery but cool
    normalized_gain = gain / max_gain

    return normalized_gain
end