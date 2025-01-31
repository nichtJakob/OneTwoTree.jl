# split node_data indices according to decision function
"""
    function split_indices(dataset::AbstractMatrix, node_data::Vector{Int64}, decision_fn::Function, decision_param::T, decision_feature::Int64) where {T<:Union{Real, String}}

Split the dataset indices contained in node_data into two sets via the decision function.

# Arguments

- `dataset::AbstractMatrix`: the dataset to split on (the datapoints are assumed to be contained rowwise)
- `node_data::Vector{Int64}`: the index list, indexing dataset, to be split
- `decision_fn::Function`: the decision function taking as input a datapoint, decision_param and decision_feature and returning a Bool
- `decision_param<:Union{Real, String}`: a parameter to the decision function. This can be a class name or a numeric threshold.
- `decision_feature::Int64`: the index of the dimension of datapoints along which to split
"""
function split_indices(dataset::AbstractMatrix, node_data::Vector{Int64}, decision_fn::Function, decision_param::T, decision_feature::Int64) where {T<:Union{Real, String}}
    true_child_data::Vector{Int64} = []
    false_child_data::Vector{Int64} = []
    for datapoint_idx in node_data
        if decision_fn(dataset[datapoint_idx, :], decision_param, feature=decision_feature)
            push!(true_child_data, datapoint_idx)
        else
            push!(false_child_data, datapoint_idx)
        end
    end
    return true_child_data, false_child_data
end

"""
    most_frequent_class(labels::Vector{T}, indices::Vector{Int64}) where T <: Union{Real, String}

Determine the most frequent class among a subset of class labels.

# Arguments

- `labels::Vector{<:Union{Real, String}}`: class labels
- `indices::Vector{Int64}`: the indices of the class labels, to be considered/counted
"""
function most_frequent_class(labels::Vector{T}, indices::Vector{Int64}) where T <: Union{Real, String}
    class_frequencies = Dict{T, Int64}()
    most_frequent = nothing

    for index in indices
        class = labels[index]
        if haskey(class_frequencies, class)
            class_frequencies[class] = class_frequencies[class] + 1
            if class_frequencies[class] > class_frequencies[most_frequent]
                most_frequent = class
            end
        else
            class_frequencies[class] = 1
            if isnothing(most_frequent)
                most_frequent = labels[index]
            end
        end
    end
    return most_frequent
end

"""
    collect_classes(dataset::AbstractMatrix, indices::Vector{Int64}, column::Int64)

Collect all unique classes among a subset of the specified column of the dataset.

# Arguments

- `dataset::AbstractMatrix`: the dataset to collect classes on
- `indices::Vector{Int64}`: the indices of the numeric labels to be considered
- `column::Int64`: the index of the dataset column/feature to collect the classes on
"""
function collect_classes(dataset::AbstractMatrix, indices::Vector{Int64}, column::Int64)
    classes = Dict{String, Bool}()
    rows = size(dataset[indices])[1]
    for i in indices
        value = dataset[i, column]
        if !haskey(classes, value)
            classes[value] = true
        end
    end
    return collect(keys(classes))
end

"""
    label_mean(labels::Vector{T}, indices::Vector{Int64}) where T<:Real

Calculate the mean of a subset of numeric labels.

- `labels::Vector{<:Real}`: the vector of numeric labels
- `indices::Vector{Int64}`: the indices of the numeric labels to be considered
"""
function label_mean(labels::Vector{T}, indices::Vector{Int64}) where T<:Real
    sum = 0.0
    foreach(index -> sum += labels[index], indices)
    return sum / size(labels[indices])[1]
end