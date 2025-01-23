
using OneTwoTree
using Test

@testset "Variance Gain Tests" begin
    @testset "Returntype" begin
        parent_labels = [1, 1, 0, 0, 1, 0]
        child_1_labels = [1, 1, 0]
        child_2_labels = [0, 1, 0]
        #println("jetzt wird vargain berrechnet")
        var_gain_result = var_gain(parent_labels, child_1_labels, child_2_labels)
        #println("var_gain_result: $var_gain_result")
        @test isa(var_gain_result, Float64)
        @test isapprox(var_gain_result, 0.5, atol=0.5)
    end

    @testset "Perfect split" begin
        parent_labels = [1, 1, 1, 0, 0, 0]
        child_1_labels = [1, 1, 1]
        child_2_labels = [0, 0, 0]
        #println("jetzt wird vargain berrechnet")
        var_gain_result = var_gain(parent_labels, child_1_labels, child_2_labels)
        #println("var_gain_result Perfect: $var_gain_result")
        @test isa(var_gain_result, Float64)
        @test isapprox(var_gain_result, 1, atol=0)
    end

    @testset "Bad split" begin
        parent_labels = [1, 0, 1, 0]
        child_1_labels = [1, 0]
        child_2_labels = [1, 0]
        #println("jetzt wird vargain berrechnet")
        var_gain_result = var_gain(parent_labels, child_1_labels, child_2_labels)
        #println("var_gain_result worst case: $var_gain_result")
        @test isa(var_gain_result, Float64)
        @test isapprox(var_gain_result, 0, atol=0)
    end
end