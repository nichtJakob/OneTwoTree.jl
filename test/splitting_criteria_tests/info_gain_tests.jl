
using OneTwoTree
using Test

@testset "Information Gain Tests" begin
    @testset "Returntype" begin
        parent_labels = [1, 1, 0, 0, 1, 0]
        child_1_labels = [1, 1, 0]
        child_2_labels = [0, 1, 0]
        #println("jetzt wird infogain berrechnet")
        info_gain_result = information_gain(parent_labels, child_1_labels, child_2_labels)
        #println("info_gain_result: $info_gain_result")
        @test isa(info_gain_result, Float64)
        @test isapprox(info_gain_result, 0.5, atol=0.5)
    end

    @testset "Perfect split" begin
        parent_labels = [1, 1, 1, 0, 0, 0]
        child_1_labels = [1, 1, 1]
        child_2_labels = [0, 0, 0]
        #println("jetzt wird infogain berrechnet")
        info_gain_result = information_gain(parent_labels, child_1_labels, child_2_labels)
        #println("info_gain_result: $info_gain_result")
        @test isa(info_gain_result, Float64)
        @test isapprox(info_gain_result, 1, atol=0)
    end

    @testset "Bad split" begin
        parent_labels = [1, 0, 1, 0]
        child_1_labels = [1, 0]
        child_2_labels = [1, 0]
        #println("jetzt wird infogain berrechnet")
        info_gain_result = information_gain(parent_labels, child_1_labels, child_2_labels)
        #println("info_gain_result: $info_gain_result")
        @test isa(info_gain_result, Float64)
        @test isapprox(info_gain_result, 0, atol=0)
    end

    @testset "Spliting relation" begin
        parent_labels = [1, 0, 1, 0, 2, 2]

        #worse split
        child_1_labels = [1, 1, 2]
        child_2_labels = [0, 0, 2]
        info_gain_result_worse = information_gain(parent_labels, child_1_labels, child_2_labels)
        @test isa(info_gain_result_worse, Float64)

        #better split
        better_1_labels = [1, 1]
        better_2_labels = [0, 0, 2, 2]
        info_gain_result_better = information_gain(parent_labels, better_1_labels, better_2_labels)
        @test isa(info_gain_result_better, Float64)

        @test info_gain_result_better > info_gain_result_worse
    end

    @testset "Char labels with swapped childs" begin
        a = 'a'
        b = 'b'
        c = 'c'
        parent_labels = [a, b, b, c, c, c]

        #split
        child_1_labels = [a, b, b]
        child_2_labels = [c, c, c]
        info_gain_result = information_gain(parent_labels, child_1_labels, child_2_labels)
        @test isa(info_gain_result, Float64)
        @test isapprox(info_gain_result, 0.5, atol=0.5)

        #same split but swaoped childs 
        info_gain_result_swapped = information_gain(parent_labels, child_2_labels, child_1_labels)
        @test isa(info_gain_result_swapped, Float64)

        @test info_gain_result == info_gain_result_swapped
    end

end