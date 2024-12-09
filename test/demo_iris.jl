#In this Example project we want to distinguish between 3 types of the Iris flower.
#using the OneTwoTree Package
using OneTwoTree

# First we load the iris dataset. Targets are here the 3 Types of the Flower
# and data contains mesurements of flowers from those types of iris
using MLDatasets
data, targets = MLDatasets.Iris()

# We split the Data in Training and test sets
train_data = data[1:120, :]
train_labels = targets[1:120]
test_data = data[121:150, :]
test_labels = targets[121:150]

# now we use the OneTwoTree Package to build a Decision-Tree
tree = DecisionTree() #TODO parameters
fit!(tree, train_data, train_labels)

# Now we can take a look at our Tree
#TODO PrintTree

#now we can classify the test set
test_predictions = predict(tree, test_data)
accuracy = sum(test_predictions .== test_labels) / length(test_labels)
println("For the Iris dataset we have achieved an Accuracy of $(accuracy)%")




