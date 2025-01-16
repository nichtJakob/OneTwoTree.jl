using Test
using OneTwoTree
# using MLDatasets    # for FashionMNIST

@testset "Load Data" begin
    features, labels = OneTwoTree.load_data("fashion_mnist_1000")
    @test typeof(labels) == Array{Int64,1}
    @test typeof(features) == Array{Float64,2}
    @test size(features) == (784, 1000)
    @test size(labels) == (1000,)
    @test labels[1] == 9
    @test features[6, 2] == 0.003921569

    let err = nothing
        try
            load_data("invalid_dataset")
        catch e
            err = e
        end
        @test err isa Exception
    end
end

# Downloading datasets works locally but not on Github Runner
# File containing the functions is in test/utils

# @testset "Download Data" begin
#     dataset_train = FashionMNIST(; split=:train)
#     let err = nothing
#         try
#             save_img_dataset_as_csv(dataset_train, "fashion_mnist_1000.csv", 1000)
#         catch e
#             err = e
#         end
#         @test err === nothing
#     end
#     path = joinpath(@__DIR__, "data", "fashion_mnist_1000.csv")
#     @test isfile(path)
# end