using DataFrames
using CSV

function load_data(name)
    datasets = ["fashion_mnist_1000"]

    if !in(name, datasets)
        error("Dataset $name not found. Possible datasets: $datasets")
    end

    data_path = joinpath(@__DIR__, "..", "..", "test", "data", "$name.csv")

    if name == "fashion_mnist_1000"
        df = DataFrame(CSV.File(data_path))
        labels = df.label
        features = Matrix(select(df, Not(:label)))
        features = transpose(features)
        return features, labels
    end
end