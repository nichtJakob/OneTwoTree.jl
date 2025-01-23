### Test utility functions used in the CART algorithm

using Test
using OneTwoTree

@testset "CART Utils" begin
    nums1 = [71.0, 58.0, 66.0, 33.0, 64.0]
    nums2 = [5.0, 69.0, 61.0, 46.0, 52.0, 100.0, 17.0]
    nums3 = [29.0, 47.0, 21.0, 22.0, 43.0, 46.0, 51.0, 65.0, 90.0, 33.0, 90.0]

    negs1 = [77.0, -86.0, -12.0, 11.0, -42.0]
    negs2 = [-50.0, -11.0, 25.0, 47.0, -70.0, -67.0, 24.0]
    negs3 = [-45.0, 18.0, 25.0, -13.0, 24.0, -9.0, 36.0, 65.0, -29.0, -44.0, -64.0]

    fracs1 = [4.55, 91.42, 56.77, 77.26, 25.66]
    fracs2 = [6.03, 85.02, 95.12, 21.36, 70.94, 25.20, 76.62]
    fracs3 = [69.99, 97.07, 53.29, 49.90, 61.08, 87.54, 73.69, 90.08, 10.12, 66.69, 24.80]

    combs1 = [66.41, -49.67, -87.17, -9.32, 88.04, 39.97, -76.66, 85.86, -30.45, 13.56, -51.52]

    inds1 = [1, 2, 3, 4, 5]
    inds2 = [2, 4, 5]
    inds3 = [5, 3, 1, 2]
    inds4 = [2, 6, 4, 5, 7]
    inds5 = [11, 3, 8, 9, 4, 1, 2]

    @testset "Subset Splitting" begin
        # TODO: Test split_indices
    end

    @testset "Class Frequency" begin
        # TODO: Test most_frequent_class
    end

    @testset "Class Collection" begin
        # TODO: Test collect_classes
    end

    @testset "Subset Mean" begin
        @test OneTwoTree.label_mean(nums1, inds3) == 64.75
        @test OneTwoTree.label_mean(nums2, inds4) == 56.8
        @test OneTwoTree.label_mean(nums3, inds5) == 52.0

        @test OneTwoTree.label_mean(negs1, inds3) == -15.75
        @test OneTwoTree.label_mean(negs2, inds4) == -15.4
        @test isapprox(OneTwoTree.label_mean(negs3, inds5), -6.1428571428571)

        @test OneTwoTree.label_mean(fracs1, inds3) == 44.6
        @test isapprox(OneTwoTree.label_mean(fracs2, inds4), 55.828)
        @test isapprox(OneTwoTree.label_mean(fracs3, inds5), 56.464285714286)

        @test isapprox(OneTwoTree.label_mean(combs1, inds5), -10.837142857143)
    end
end