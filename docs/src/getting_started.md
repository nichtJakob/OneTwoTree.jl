## 🛠️ Prerequisites

| Prerequisite | Version | Installation Guide | Required |
|--------------|---------|--------------------|----------|
| Julia       | 1.10    | [![Julia](https://img.shields.io/badge/Julia-v1.10-blue)](https://julialang.org/downloads/) | ✅ |

## 🚀 Getting Started

#### ✨ Downloading the Package
- Via `Pkg>` mode (press `]` in Julia REPL):

```bash
activate --temp
add https://github.com/nichtJakob/OneTwoTree.jl.git
```

- For Pluto notebooks: We can't use Pluto's environments but have to create our own:
```julia
using Pkg
Pkg.activate("MyEnvironment")
Pkg.add(url="https://github.com/nichtJakob/OneTwoTree.jl.git")
using OneTwoTree
```


## ▶️ **Example**

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
print("The tree predicted class \$(prediction[1]).")
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
print("The tree predicted \$(prediction[1]).")
```

### Forests and Loading Other Datasets

You can find more extensive examples utilising the `Iris` and `BostonHousing` datasets from `MLDatasets` in [`demo_classification.jl`](https://github.com/nichtJakob/OneTwoTree.jl/blob/master/demo_classification.jl). and [`demo_regression.jl`](https://github.com/nichtJakob/OneTwoTree.jl/blob/master/demo_regression.jl). The latter further compares `DecisionTree` performance to that of a `Forest`.

## 📚 Further Reading for Developers


- ✨ **Downloading the Code for Local Development**

```bash
git clone https://github.com/nichtJakob/OneTwoTree.jl.git
```

- 🔧 **Installation and Dependency Setup**

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
     include("trees_tests/regression_tests.jl") # run specific test (example)
     ```

    For a quick guide on how to develop julia packages, write tests, ...,  read [this](https://adrianhill.de/julia-ml-course/write/).


  ## 🏙️ Project Structure
  Here is an overview of the project's main components:

  - `src/`: Contains code for the main functionality of the package.
    - `OneTwoTree.jl`: Main entry point of the project. Contains includes and exports.
    - `splitting_criteria/`: Splitting Criteria to construct trees from datasets.
      - `gini.jl`: Gini Impurity
      - `info_gain`: Information Gain
      - `var_gain.jl`: Variance Gain
    - `trees/`
      - `cart/`: CART algorithm for constructing trees from data.
      - `tree.jl`: Classification and regression trees.
      - `node.jl`: One node of a decision tree.
      - `forest.jl`: Classification and regression forests for aggregating the decisions of multiple decision trees.
      - `decision_function.jl`: Decision of a node, e.g. `is x[1] > 5 ?`
    - `utils/`: Functions we didn't want to import a dependency for.
  - `test/`: Unit tests to ensure correctness of various package components.
  - `docs/`: Files for the external documentation (which you are currently reading :D).

