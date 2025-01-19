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

# Spliting criterion for regression based on the variance similar to infogain
function var_gain(parent_labels::AbstractVector, child_1_labels::AbstractVector, child_2_labels::AbstractVector) :: Float64
    max_gain = variance(parent_labels)

    if isempty(child_1_labels) || isempty(child_2_labels) || max_gain == 0.0
        return 0.0
    end

    total = length(parent_labels) 

    child_1_weight = length(child_1_labels) / total
    child_2_weight = length(child_2_labels) / total

    weighted_var_1 = child_1_weight * variance(child_1_labels) 
    weighted_var_2 = child_2_weight * variance(child_2_labels) 
    weighted_var = weighted_var_1 + weighted_var_2

    gain = variance(parent_labels) - weighted_var

    # normalization not neccessery but cool
    normalized_gain = gain / max_gain

    return normalized_gain
end