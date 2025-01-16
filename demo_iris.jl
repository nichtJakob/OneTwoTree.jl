#In this Example project we want to distinguish between 3 types of the iris flower
#using the OneTwoTree Package
using OneTwoTree

# If you want to execute this example in your julia REPL you will first need to
# 1) install a dependency for the dataset: Pkg.add("MLDatasets")
# 2) execute the code in your REPL: include("demo_iris.jl")

# First we load the iris dataset. Targets are the 3 types of the flowers
# and data contains measurements of flowers
using MLDatasets

dataset = MLDatasets.Iris()
data = Array(dataset.features)
targets = String.(Array(dataset.targets))

#lets have a look at the types of the iris flower contained in targets
println("The possible targets are: ", unique(targets), "\n")
# and the features aka the given measurmentrs of flowers 
println("The measured features are: ", names(dataset.features), "\n")

# we have 150 data points as you can see in the size of the data 
println("Size of data: ", size(data), "\n")

#here we define how big training and test set should be
# you can modify the spliting point to be any value between 1 and 149
spliting_point = 120
println("It is $(spliting_point > 1 && spliting_point < 150) that you have chosen a qualifiing spliting point $(spliting_point)", "\n")



# We split the Data in Training and test sets
train_data = data[1:spliting_point, :]
train_labels = targets[1:spliting_point]
test_data = data[spliting_point+1:150, :]
test_labels = targets[spliting_point+1:150]

println("Size of Train data: ", size(train_data), "\n")
println("Size of Test data: ", size(test_data), "\n")

# now we use the OneTwoTree Package to build a Decision-Tree, you can experiment with diferent tree-depths
our_max_depth = 3
tree = DecisionTreeClassifier(max_depth=our_max_depth)

# now we train on the training data
fit!(tree, train_data, train_labels)

# finaly we can take a look at our trained Tree
println("\n \n Our Tree: \n")
print_tree(tree)

#Lets see how good our tree is at predicting lables of datapoints in the test_data
test_predictions = predict(tree, test_data)
accuracy = sum(test_predictions .== test_labels) / length(test_labels)

println("\n\nFor the Iris dataset we have achieved a test-accuracy of $(round(accuracy * 100, digits=2))%")