"""
    Decision

A structure representing a decision with a function and its parameter.

# Parameters
- `fn::Function`: The decision function.
- `param::Union{Number, String}`: The parameter for the decision function.
    - Number: for comparison functions (e.g. x < 5.0)
    - String: for True/False functions (e.g. x == "red" or x != 681)
- `feature::Int64`: The index of the feature to compare.
"""
struct Decision{S<:Union{Number, String}}
    fn::Function
    param::S
    feature::Int64

    function Decision(fn::Function, feature::Int64, param::S) where S
        # TODO: feature index can be chosen out of bounds... Idk, just be careful?
        new{S}(fn, param, feature)
    end
end

# for easier calls depending on data type

function call(decision::Decision, datapoint::Vector{S}) where S
    if length(datapoint) < decision.feature
        error("call: passed datapoint of insufficient dimensionality!")
    end
    return decision.fn(datapoint, decision.param, feature=decision.feature)
end

function call(decision::Decision, dataset::Matrix{S}) where S
    if size(dataset, 2) < decision.feature
        error("call: passed dataset with data of insufficient dimensionality!")
    end
    return [decision.fn(datapoint, decision.param, feature=decision.feature) for datapoint in dataset]
end

#--------------------------------------
# MARK: Printing
#--------------------------------------

"""
    _decision_to_string(d::DecisionFn)

Returns a string representation of the decision function.

# Arguments
- `d::DecisionFn`: The decision function to convert to a string.
"""
function _decision_to_string(d::Decision)
    if isa(d.param, Number)
        return "x[" * string(d.feature) * "] <= " * string(d.param)
    else
        return "x[" * string(d.feature) * "] == " * string(d.param)
    end
end

function Base.show(io::IO, d::Decision)
    print(io, _decision_to_string(d))
end

#--------------------------------------
#MARK: Decision Functions
#--------------------------------------

"""
    less_than_or_equal

A basic numerical decision function for testing and playing around.
"""
function less_than_or_equal(x, threshold::Float64; feature::Int64 = 1)::Bool
    return x[feature] <= threshold
end

"""
    equal

A basic categorical decision function for testing and playing around.
"""
function equal(x, class::String; feature::Int64 = 1)::Bool
    return x[feature] == class
end

