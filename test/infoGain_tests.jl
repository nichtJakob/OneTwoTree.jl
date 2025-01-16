
using OneTwoTree
using Test

@testset "Information Gain Tests" begin
    @testset "Returntype" begin
        parent_labels = [1, 1, 0, 0, 1, 0]
        child_1_labels = [1, 1, 0]
        child_2_labels = [0, 1, 0]
        println("jetzt wird infogain berrechnet")
        info_gain_result = information_gain(parent_labels, child_1_labels, child_2_labels)
        println("info_gain_result: $info_gain_result")
        @test isa(info_gain_result, Float64)
        @test isapprox(info_gain_result, 0.5, atol=3)
    end

    @testset "Perfect split" begin
        parent_labels = [1, 1, 1, 0, 0, 0]
        child_1_labels = [1, 1, 1]
        child_2_labels = [0, 0, 0]
        println("jetzt wird infogain berrechnet")
        info_gain_result = information_gain(parent_labels, child_1_labels, child_2_labels)
        println("info_gain_result: $info_gain_result")
        @test isa(info_gain_result, Float64)
        @test isapprox(info_gain_result, 1, atol=0)
    end

    @testset "Bad split" begin
        parent_labels = [1, 0, 1, 0]
        child_1_labels = [1, 0]
        child_2_labels = [1, 0]
        println("jetzt wird infogain berrechnet")
        info_gain_result = information_gain(parent_labels, child_1_labels, child_2_labels)
        println("info_gain_result: $info_gain_result")
        @test isa(info_gain_result, Float64)
        @test isapprox(info_gain_result, 0, atol=0)
    end
end