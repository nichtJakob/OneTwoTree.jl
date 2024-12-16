using DataFrames
using CSV

"""
    load_data(name::String)

Load a preconfigured dataset from a CSV file.

# Arguments
- `name::String`: the name of the dataset to load
# Example
`load_data("fashion_mnist_1000")`
"""
function load_data(name)
    datasets = ["fashion_mnist_1000"]

    if !in(name, datasets)
        error("Dataset $name not found. Possible datasets: $datasets.")
    end

    data_path = joinpath(@__DIR__, "..", "..", "test", "data", "$name.csv")

    if name == "fashion_mnist_1000"
        df = DataFrame(CSV.File(data_path))
        labels = df.label
        features = Matrix(select(df, Not(:label)))
        features = Array(transpose(features))
        return features, labels
    end

    error("logic error: dataset $name is not handled correctly.")
end