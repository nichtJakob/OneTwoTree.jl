"""
    split(N)

Determine the optimal split of a node. Handles numerical and categorical data and labels.

# Arguments

- `N::Node`: The node to be split. All additional information for the split calculation (e.g. dataset, labels, node_data) is contained in N.
"""
function split(N::Node)
    decision::Union{Decision, Nothing} = nothing

    # 1. find best feature to split i.e. calc best split for each feature
    num_features = size(N.dataset)[2]
    best_feature = -1
    best_decision::Union{Decision, Nothing} = nothing
    best_impurity = -1.0

    data = N.dataset[N.node_data, :]
    # @info "\n\nUsing data" data
    for i in range(1, num_features)
        # NOTE: This determination of whether a column is categorical or numerical assumes, that the types do not vary among a column
        is_categorical = (typeof(N.dataset[1, i]) == String)
        # @info "\n\n\nChecking decisions for feature $(i) where is_categorical=$(is_categorical): "
        # for categorical features, we calculate the gini impurity for each split (e.g. feature == class1, feature == class2, ...)
        if is_categorical
            # TODO: Test & Debug Categorical case
            classes = collect_classes(N.dataset, N.node_data, i)
            if size(classes)[1] >= 2
                for class in classes

                    decision = Decision(equal, i, class)
                    if N.classify
                        impurity = gini_impurity(N.dataset, N.labels, N.node_data, decision.fn, decision.param, decision.feature)
                    else
                        impurity = variance(N.dataset, N.labels, N.node_data, decision.fn, decision.param, decision.feature)
                    end

                    if best_feature == -1 || (impurity < best_impurity)
                        best_feature = i
                        best_impurity = impurity
                        best_decision = decision
                    end
                end
            end
        # for numerical features, we sort them and calculate the gini impurity for each split (splitting at the mean between each two list neighbors)
        else
            # sort dataset matrix by column
            feature_value_sorting = sortperm(data[:, i])
            j = 1
            while j < length(feature_value_sorting)
                value = data[feature_value_sorting[j], i]
                next_value = data[feature_value_sorting[j+1], i]
                # if next_value == value there is no discriminating decision,
                # thus we forward to the next distinct value
                while next_value == value
                    j += 1
                    if j < size(feature_value_sorting)[1]
                        next_value = data[feature_value_sorting[j+1], i]
                    else
                        j = -1
                        break
                    end
                end
                if j == -1
                    break
                end
                # calculate threshold used to discriminate between two values
                midpoint = (value + next_value)/2.0

                # calculate splitting impurity
                decision = Decision(less_than_or_equal, i, midpoint)
                if N.classify
                    impurity = gini_impurity(N.dataset, N.labels, N.node_data, decision.fn, decision.param, decision.feature)
                else
                    impurity = variance(N.dataset, N.labels, N.node_data, decision.fn, decision.param, decision.feature)
                end

                # check if we found an improving decision
                if best_feature == -1 || (impurity < best_impurity)
                    best_feature = i
                    best_impurity = impurity
                    best_decision = decision
                end
                j += 1
            end
        end
    end

    # @info "determined best decision as " best_decision best_impurity
    # if best_decision == nothing, this means that no split could be found.
    return best_decision, best_impurity
end

"""
    should_split(N, post_split_impurity, max_depth)

Determines whether to split the node N given.

# Arguments

- `N::Node`: Node that may be split. N contains further fields relevant to the decision like the best splitting decision function, it's leaf impurity and depth.
- `post_split_impurity::Float64`: The impurity of N after it's optimal split.
- `max_depth::Int64`: The maximum depth of the tree N is part of.
"""
function should_split(N::Node, post_split_impurity::Float64, max_depth::Int64)
    # TODO: implement actual splitting decision logic i.e. do we want to split this node yey or nay?
    # There are a variety of criteria one could imagine. For now we only posit that the current node should be impure i.e. impurity > 0 and the max_depth hasn't been reached.
    if N.decision === nothing || post_split_impurity == -1.0
        # @info "Could not find optimal split => No Split"
        return false
    end
    if N.impurity == 0.0
        # @info "Node impurity == 0.0 => No Split"
        return false
    end
    if N.depth == max_depth
        # @info "max_depth has been reached => No Split"
      return false
    end
    # if N.impurity - post_split_impurity < min_purity_gain
    #   return false
    # end
    return true
end