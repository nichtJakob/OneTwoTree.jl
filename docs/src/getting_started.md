## 🚀 Getting Started

#### Explanation of Folders and Files 
Explanation of Folders and Files
`docs/`
  - Contains everything related to documentation.
  - `make.jl`: Script to build the documentation using Documenter.jl.
  - `Project.toml` & Manifest.toml: Separate environment for documentation dependencies.
  - `src/`: Markdown files structured for different sections of the documentation.
  - `index.md`: Main landing page for the documentation.
  - `functions.md`: Lists functions and usage.
  - `examples/`: Specific examples for classification and regression use cases.
  - `api/`: Detailed API reference for classifiers, regressors, and utility functions.
  - `assets/`: Store images or other resources for the documentation.
`build/`
  - Contains the generated documentation output (ignored by Git).
`src/`
  - The main Julia module for OneTwoTree resides here. Documentation and code examples will often reference these files.
`test/`
  - Unit tests ensure code correctness and complement examples in the documentation.
`README.md`

A concise overview of the project for GitHub visitors. It should link to the generated documentation hosted (e.g., on GitHub Pages).


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


## ▶️ **Example: Running a Simple Example**

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
     include("trees_tests/regression_tests.jl") # run specific test (example)
     ```

    For a quick guide on how to develop julia packages, write tests, ...,  read [this](https://adrianhill.de/julia-ml-course/write/).

## 👩‍💻 Contributors
[![Contributors](https://contrib.rocks/image?repo=nichtJakob/OneTwoTree.jl)](https://github.com/nichtJakob/OneTwoTree.jl/graphs/contributors)

# OneTwoTree