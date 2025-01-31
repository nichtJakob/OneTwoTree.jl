using OneTwoTree
using Test

@testset "call tests" begin
    #@test false
    decision = OneTwoTree.Decision(OneTwoTree.equal, 3, "A")
    
    datapoint = [4.0, 6.0]
    @test_throws ArgumentError OneTwoTree.call(decision, datapoint)

    dataset = [1 2; 3 4]
    @test_throws ArgumentError OneTwoTree.call(decision, dataset)

    decision2 = OneTwoTree.Decision(OneTwoTree.equal, 1, "A")
    @test OneTwoTree.call(decision2, dataset) == Bool[0 0; 0 0]

    decision_ltoe = OneTwoTree.Decision(OneTwoTree.less_than_or_equal, 18, 7.76)
    @test OneTwoTree._decision_to_string(decision_ltoe) == "x[18] <= 7.76"

    decision_e = OneTwoTree.Decision(OneTwoTree.equal, 42, "yes")
    @test OneTwoTree._decision_to_string(decision_e) == "x[42] == yes"
end