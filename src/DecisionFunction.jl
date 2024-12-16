"""
    DecisionFunction

A structure representing a decision with a function and its parameter.

# Parameters
- `fn::Function`: The decision function.
- `param::Union{Number, String}`: The parameter for the decision function.
    - Number: for comparison functions (e.g. x < 5.0)
    - String: for True/False functions (e.g. x == "red" or x != 681)
"""
struct DecisionFunction
    fn::Function
    param::Union{Number, String}
end


function Base.show(io::IO, fn::DecisionFunction)
    print(io, DecisionFn_to_string(fn))
end

"""
    DecisionFn_to_string(d::DecisionFn)

Returns a string representation of the decision function.

# Arguments
- `d::DecisionFn`: The decision function to convert to a string.
"""
function DecisionFn_to_string(d::DecisionFunction)
    if isa(d.param, Number)
        return "x < " * string(d.param)
    else
        return "x " * string(d.param)
    end
end
