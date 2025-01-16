[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nichtJakob.github.io/OneTwoTree.jl/dev/)
[![Build Status](https://github.com/nichtJakob/OneTwoTree.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/nichtJakob/OneTwoTree.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/nichtJakob/OneTwoTree.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/nichtJakob/OneTwoTree.jl)

<div align="center">
  <img src="assets/decision_tree_logo.svg" height="256" />
  <h1>OneTwoTree</h1>
  <p>Julia Package implementing Decision Trees and Random Forests for Machine Learning.</p>
</div>

## Brief Explanation

[Decision Trees](https://en.wikipedia.org/wiki/Decision_tree) are a supervised learning algorithm used for classification and regression tasks. They split the data into subsets based on feature values, forming a tree-like structure where each internal node represents a decision based on a feature, and each leaf node represents a predicted outcome.

[Random Forests](https://en.wikipedia.org/wiki/Random_forest) improve on Decision Trees by creating an ensemble of multiple decision trees, each trained on a random subset of the data. The final prediction is made by averaging the outputs of all trees (for regression) or using a majority vote (for classification), which helps reduce overfitting and improves model accuracy.

## 🛠️ Prerequisites

| Prerequisite | Version | Installation Guide | Required |
|--------------|---------|--------------------|----------|
| Julia       | 1.10    | [![Julia](https://img.shields.io/badge/Julia-v1.10-blue)](https://julialang.org/downloads/) | ✅ |


## 🚀 Getting Started

#### ✨ Downloading the Package
- Via `Pkg>` mode (press `]` in Julia REPL):

```bash
add https://github.com/nichtJakob/OneTwoTree.jl
```

- For Pluto notebooks: We can't use Pluto's environments but have to create our own:
```julia
using Pkg
Pkg.activate("MyEnvironment")
Pkg.add(url="https://github.com/nichtJakob/OneTwoTree.jl")
using OneTwoTree
```


## ▶️ **Example: Running a Simple Example**

- Note that the classifier currently only supports training datasets of type `Real` and labels of type `String`
- Note that that the Tree Construction in its current state can be very slow. Therefore, it may be advised to use small training datasets for the moment.

### Classification
  ```julia
  using OneTwoTree
  dataset = [ # The rows are the different data points
    3.5 9.1 2.9
    1.0 1.2 0.4
    5.6 3.3 4.3
  ]
  labels = ["A", "B", "C"]

  tree = DecisionTreeClassifier(max_depth=2)
  fit!(tree, dataset, labels) # train the tree with the data
  print(tree)

  prediction = predict(tree, [
    2.0 4.0 6.0
  ])
  print("The tree predicted class $(prediction[1]).")
  ```

### Regression
  ```julia
  using OneTwoTree
  dataset = [
    1.0 2.0
    2.0 3.0
    3.0 4.0
    4.0 5.0
  ]
  labels = [1.5, 2.5, 3.5, 4.5]

  tree = DecisionTreeRegressor(max_depth=3)
  fit!(tree, dataset, labels)
  print(tree)

  prediction = predict(tree, [
    1.0 4.0
  ])
  print("The tree predicted $(prediction[1]).")
  ```
   ### Loading Other Datasets
   You can find a more extensive example which utilises the `Iris` dataset from `MLDatasets` in [`demo_iris.jl`](https://github.com/nichtJakob/OneTwoTree.jl/blob/master/demo_iris.jl). :)

## 📚 **Further Reading for Developers**


1. ✨ **Downloading the Code for Local Development**

      ``` bash
      git clone https://github.com/nichtJakob/OneTwoTree.jl.git
      ```




2. 🔧 **Installation and Dependency Setup**

    - Run the following commands in the package's root directory to install the dependencies and activate the package's virtual environment:

      ```bash
      julia --project
      ```
    - It might be necessary to resolve dependencies.
    Go into `Pkg>` mode by pressing `]`. Then type
      ```julia
      resolve
      ```
   - To execute the tests, type in `Pkg>` mode:
     ```julia
     test
     ```

     or in your julia REPL run:
     ```julia
     include("runtests.jl")         # run all tests
     include("regression_tests.jl") # run specific test (example)
     ```

    For a quick guide on how to develop julia packages, write tests, ...,  read [this](https://adrianhill.de/julia-ml-course/write/).

## 👩‍💻 Contributors
[![Contributors](https://contrib.rocks/image?repo=nichtJakob/OneTwoTree.jl)](https://github.com/nichtJakob/OneTwoTree.jl/graphs/contributors)
