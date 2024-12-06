using Test
using Flux
include("../src/gini.jl")
using .Gini

@testset "Gini.jl" begin
    @testset "Gini" begin
        features1 = [true, false, true, true, false]
        labels1 = [true, false, true, false, false]
        decision_fn1 = x -> x == true
        gini1 = gini_impurity(features1, labels1, decision_fn1)
        @test isapprox(gini1, 0.266, atol=1e-2)
    end
end

