# Test extensions to basic Base methods
@testset "Base Extensions" begin
    # initialize models and references
    m = InfiniteModel()
    m2 = InfiniteModel()
    ivref = InfOptVariableRef(m, 1, Infinite)
    pvref = InfOptVariableRef(m, 2, Point)
    gvref = InfOptVariableRef(m, 3, Global)
    pref = InfOptVariableRef(m, 1, Parameter)
    # variable compare
    @testset "(==)" begin
        @test ivref == ivref
        @test pvref == pvref
        @test gvref == gvref
        @test ivref == InfOptVariableRef(m, 1, Infinite)
        @test pvref == InfOptVariableRef(m, 2, Point)
        @test gvref == InfOptVariableRef(m, 3, Global)
        @test !(ivref == InfOptVariableRef(m, 2, Infinite))
        @test !(ivref == InfOptVariableRef(m2, 1, Infinite))
        @test !(ivref == InfOptVariableRef(m, 1, Global))
        @test !(ivref != InfOptVariableRef(m, 1, Infinite))
        @test !(pref == ivref)
        @test !(ivref == pvref)
    end
    # copy(v)
    @testset "copy(v)" begin
        @test copy(ivref) == ivref
        @test copy(pvref) == pvref
        @test copy(gvref) == gvref
    end
    # copy(v, m)
    @testset "copy(v, m)" begin
        @test copy(ivref, m2) == InfOptVariableRef(m2, 1, Infinite)
        @test copy(pvref, m2) == InfOptVariableRef(m2, 2, Point)
        @test copy(gvref, m2) == InfOptVariableRef(m2, 3, Global)
    end
    # broadcastable
    @testset "broadcastable" begin
        @test isa(Base.broadcastable(ivref), Base.RefValue{InfOptVariableRef})
        @test isa(Base.broadcastable(pvref), Base.RefValue{InfOptVariableRef})
        @test isa(Base.broadcastable(gvref), Base.RefValue{InfOptVariableRef})
    end
end

# Test core JuMP methods
@testset "Core JuMP Extensions" begin
    # initialize models and references
    m = InfiniteModel()
    m2 = InfiniteModel()
    ivref = InfOptVariableRef(m, 1, Infinite)
    pvref = InfOptVariableRef(m, 2, Point)
    gvref = InfOptVariableRef(m, 3, Global)
    pref = InfOptVariableRef(m, 1, Parameter)
    # isequal_canonical
    @testset "JuMP.isequal_canonical" begin
        @test isequal_canonical(ivref, ivref)
        @test isequal_canonical(pvref, pvref)
        @test isequal_canonical(gvref, gvref)
        @test !isequal_canonical(ivref, InfOptVariableRef(m2, 1, Infinite))
        @test !isequal_canonical(ivref, InfOptVariableRef(m, 2, Infinite))
    end
    # variable_type(m)
    @testset "JuMP.variable_type(m)" begin
        @test variable_type(m) == GeneralVariableRef
    end
    # variable_type(m, t)
    @testset "JuMP.variable_type(m, t)" begin
        @test JuMP.variable_type(m, Infinite) == InfOptVariableRef
        @test JuMP.variable_type(m, Point) == InfOptVariableRef
        @test JuMP.variable_type(m, Global) == InfOptVariableRef
        @test JuMP.variable_type(m, Parameter) == InfOptVariableRef
        @test_throws ErrorException JuMP.variable_type(m, Reduced)
        @test_throws ErrorException JuMP.variable_type(m, Measure)
        @test_throws ErrorException JuMP.variable_type(m, :bad)
    end
end

# Test precursor functions needed for add_parameter
@testset "Basic Reference Queries" begin
    # initialize model and infinite variable
    m = InfiniteModel()
    ivref = InfOptVariableRef(m, 1, Infinite)
    info = VariableInfo(false, 0, false, 0, false, 0, false, 0, false, false)
    param = InfOptParameter(IntervalSet(0, 1), Number[], false)
    pref = add_parameter(m, param, "test")
    m.vars[1] = InfiniteVariable(info, (pref, ))
    # JuMP.index
    @testset "JuMP.index" begin
        @test JuMP.index(ivref) == 1
    end
    # JuMP.owner_model
    @testset "JuMP.owner_model" begin
        @test owner_model(ivref) == m
    end
    # JuMP.is_valid
    @testset "JuMP.is_valid" begin
        @test is_valid(m, ivref)
        @test !is_valid(InfiniteModel(), ivref)
        @test !is_valid(m, InfOptVariableRef(m, 5, Infinite))
    end
end
