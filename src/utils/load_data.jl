

function load_data(name)
    datasets = ["fashion_mnist_1000"]

    if !in(name, datasets)
        error("Dataset $name not found. Possible datasets: $datasets")
    end

    data_path = joinpath(dirname(pathof(OneTwoTree)), "..", "test/data/", "$name.csv")

    if name == "fashion_mnist_1000"
        df = CSV.read(path)
    end
end