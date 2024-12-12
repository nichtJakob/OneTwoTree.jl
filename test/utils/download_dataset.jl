using MLDatasets
using DataFrames
using CSV

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

function save_img_dataset_as_csv(dataset, filename, num_samples)
    # x, y = dataset_train[1] # one image

    images, labels = dataset[1:num_samples]
    df = convert_img_to_dataframe(images, labels)

    CSV.write(filename, df)
    println("Saved $(length(labels)) samples to \"$filename\"")
end

dataset_train = FashionMNIST(; split=:train)
save_img_dataset_as_csv(dataset_train, "test/data/fashion_mnist_1000.csv", 1000)
