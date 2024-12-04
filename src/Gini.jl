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

function gini_impurity(features::AbstractVector, labels::Vector{Bool}, decision_fn::Function)::Float64

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
    if length(labels) == 0

        return 0
    else
        #Number of true in true_labels and false_labels
        true_num_true = count(x -> x == true, true_labels)
        false_num_true = count(x -> x == true, false_labels)

        #Number of false in true_labels and false_labels
        true_num_false = count(x -> x == false, true_labels)
        false_num_false = count(x -> x == false, false_labels)

        #Calculate proportions
        total_true = length(true_labels)
        total_false = length(false_labels)

        #Gini for true nod
        gini_true = 1 - (true_num_true/total_true)^2 - (true_num_false/total_true)^2

        #Gini for false nod
        gini_false = 1 - (false_num_true/total_false)^2 - (false_num_false/total_false)^2

        #weighted gini
        total_length_data = length(features)
        gini_total = length(split_true)/total_length_data * gini_true + length(split_false)/total_length_data * gini_false

        return gini_total

    end
end 

features = [true, false, true, true, false]
labels = [true, false, true, false, false]
decision_fn = x -> x == true

gini = gini_impurity(features, labels, decision_fn)
println("Gini Impurity: $gini")


features = [25, 40, 35, 22, 60]  # Ages as the features
labels = [true, false, true, false, true]  # Whether they love "Cool as Ice"
decision_fn = (x -> x > 30)  # Decision function: split based on age > 30

# Calculate Gini impurity
gini = gini_impurity(features, labels, decision_fn)
println("Gini Impurity: $gini")