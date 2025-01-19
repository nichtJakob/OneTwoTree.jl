```@meta
CurrentModule = OneTwoTree
```

# OneTwoTree

Documentation for [OneTwoTree](https://github.com/nichtJakob/OneTwoTree.jl).

@index

## Core Structures
```@docs
OneTwoTree.Decision
OneTwoTree.DecisionTreeClassifier
OneTwoTree.DecisionTreeRegressor
OneTwoTree.Node
```

## Training and Prediction
```@docs

OneTwoTree.fit!
OneTwoTree.predict
OneTwoTree.should_split
OneTwoTree.split
```

## Utility Functions
```@docs

OneTwoTree.calc_accuracy
OneTwoTree.calc_depth
OneTwoTree.collect_classes
OneTwoTree.most_frequent_class
OneTwoTree.print_tree
```

## Internal Methods
```@docs

OneTwoTree._decision_to_string
OneTwoTree._node_to_string
OneTwoTree._tree_to_string
OneTwoTree._verify_fit!_args
```

## Rest
```@docs
OneTwoTree.entropy
OneTwoTree.information_gain
OneTwoTree.label_mean
OneTwoTree.equal
OneTwoTree.split_indices
OneTwoTree.less_than_or_equal
OneTwoTree.load_data
OneTwoTree.is_leaf
OneTwoTree.printM
```
