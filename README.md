
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://nichtJakob.github.io/OneTwoTree.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nichtJakob.github.io/OneTwoTree.jl/dev/)
[![Build Status](https://github.com/nichtJakob/OneTwoTree.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/nichtJakob/OneTwoTree.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/nichtJakob/OneTwoTree.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/nichtJakob/OneTwoTree.jl)

<div align="center">
  <img src="assets/decision_tree_logo.svg" height="256" />
  <h1>OneTwoTree</h1>
  <p>Julia Package implementing Decision Trees and Random Forests for Machine Learning.</p>
</div>

## 🛠️ Prerequisites

| Prerequisite | Version | Installation Guide | Required |
|--------------|---------|--------------------|----------|
| Julia       | 1.10    | [![Julia](https://img.shields.io/badge/Julia-v1.10-blue)](https://julialang.org/downloads/) | ✅ |


## 🚀 Getting Started

1. ✨ **Downloading the Code**

    To get started, download the code using one of the following links:

  - **Normal HTTPS Link:**

      ``` bash
      git clone https://github.com/nichtJakob/OneTwoTree.jl.git
      ```


  - **SSH Link:**

      ``` bash
      git clone git@github.com:nichtJakob/OneTwoTree.jl.git
      ```


2. 🔧 **Installation and Dependency Setup**

    Run the following commands in the package's root directory to install the dependencies and activate the package's virtual environment:

  - For Contributors:

      ```bash
      julia --project
      ```
    It might be necessary to resolve dependencies.
    Go into the package manager by pressing `]`. Then type
      ```julia
      resolve
      ```


3. ▶️ **Example: Running a Simple Example**

    ### Classification
    ```julia
    using OneTwoTree

    # The rows are the different data points
    dataset = [
      3.5 9.1 2.9
      1.0 1.2 0.4
      5.6 3.3 4.3
    ]
    labels = ["A", "B", "C"]

    tree = DecisionTreeClassifier(max_depth=2)
    fit!(tree, dataset, labels)
    print(tree)

    prediction = predict(tree, [
      2.0 4.0 6.0
    ])
    print("The tree predicted class $(prediction[1]).")
    ```
    - Note that the classifier currently only supports training datasets of type `Real` and labels of type `String`
    - Note that that the Tree Construction in its current state can be very slow. Therefore, it may be advised to use small training datasets for the moment.

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
   ### Loading Datasets
   You can find a more extensive example which utilises the `Iris` dataset from `MLDatasets` in `demo_iris.jl`. :)

5. 📚 **Further Reading**

    If you are a contributor to this package, read [this](https://adrianhill.de/julia-ml-course/write/) for information on how to add code, write tests, add dependencies, etc.

## 👩‍💻 Contributors
[![Contributors](https://contrib.rocks/image?repo=nichtJakob/OneTwoTree.jl)](https://github.com/nichtJakob/OneTwoTree.jl/graphs/contributors)
