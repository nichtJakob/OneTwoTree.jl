using Test
using OneTwoTree

@testset "Load Data" begin
    features, labels = load_data("fashion_mnist_1000")
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