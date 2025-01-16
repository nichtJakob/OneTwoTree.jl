using MLDatasets
using DataFrames
using CSV

"""
    convert_img_to_dataframe(images, labels)

Convert an image dataset to a DataFrame which we need to write to CSV.

# Arguments
- `images::Array{T,3}`: the images, dimensions (width, height, batch_size)
- `labels::Array{S,1}`: the labels
"""
function convert_img_to_dataframe(images, labels)
    w, h, batch_size = size(images)
    images_flat = reshape(images, w * h, :)

    # put every feature into separate column
    feature_names = ["pixel_$i" for i in 1:w*h]
    images_flat = transpose(images_flat)
    df_images = DataFrame(images_flat, feature_names)

    # concat
    df_labels = DataFrame(label=labels)
    df_final = hcat(df_labels, df_images)

    return df_final
end

"""
    save_img_dataset_as_csv(dataset, filename, num_samples)

Save a subset of an image dataset to a CSV file.

# Arguments
- `dataset`: the dataset to save, e.g. FashionMNIST(; split=:train)
- `filename::String`: the name of the file to save (not the entire path)
- `num_samples::Int`: the maximum number of samples to save from the dataset
"""
function save_img_dataset_as_csv(dataset, filename, num_samples)

    save_path = joinpath(@__DIR__, "..", "data")
    if !isdir(save_path)
        error("Directory \"$save_path$filename\" does not exist.")
    end
    save_path = joinpath(save_path, filename)

    if length(dataset) < num_samples
        num_samples = length(dataset)
    end

    images, labels = dataset[1:num_samples]
    df = convert_img_to_dataframe(images, labels)

    CSV.write(save_path, df)
    println("Saved $(length(labels)) samples to \"$save_path\"")
end

#dataset_train = FashionMNIST(; split=:train)
#save_img_dataset_as_csv(dataset_train, "fashion_mnist_1000.csv", 1000)
