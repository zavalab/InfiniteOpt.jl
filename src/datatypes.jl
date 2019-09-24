# Define variable and union types
const Infinite = :Infinite # this is for both variable and constraint refs
const Point = :Point
const Global = :Global
const Parameter = :Parameter
const MeasureRef = :MeasureRef # this is for both variable and constraint refs
const Reduced = :Reduced
const Variables = Union{Val{Infinite}, Val{Point}, Val{Global}}
const Finite = :Finite # this is for constraint refs only
const VariableRef = :VariableRef # this denotes JuMP.VariableRef type

"""
    AbstractInfiniteSet

An abstract type for sets that characterize infinite parameters.
"""
abstract type AbstractInfiniteSet end

"""
    InfOptParameter{T <: AbstractInfiniteSet} <: JuMP.AbstractVariable

A DataType for storing core infinite parameter information.

**Fields**
- `set::T` The infinite set that characterizes the parameter.
- `supports::Vector{<:Number}` The support points used to discretize this
                               parameter.
- `independent::Bool` Is independent of other parameters that share its group ID
                      number.
"""
struct InfOptParameter{T <: AbstractInfiniteSet} <: JuMP.AbstractVariable
    set::T
    supports::Vector{<:Number}
    independent::Bool
end

# Extend to handle InfOptParameters correctly
function Base.:(==)(p1::InfOptParameter, p2::InfOptParameter)
    check1 = p1.set == p2.set
    check2 = isequal(p1.supports, p2.supports)
    check3 = p1.independent == p2.independent
    return (check1 && check2 && check3)
end

"""
    InfOptVariable <: JuMP.AbstractVariable

An abstract type for infinite, point, and global variables.
"""
abstract type InfOptVariable <: JuMP.AbstractVariable end

"""
    AbstractReducedInfo

An abstract type reduced variable information.
"""
abstract type AbstractReducedInfo end

"""
    AbstractMeasureData

An abstract type to define data for measures to define the behavior of
[`Measure`](@ref).
"""
abstract type AbstractMeasureData end

"""
    Measure{T <: JuMP.AbstractJuMPScalar, V <: AbstractMeasureData}

A DataType for measure abstractions.

**Fields**
- `func::T` Infinite variable expression.
- `data::V` Data of the abstraction as described in a `AbstractMeasureData`
            subtype.
"""
struct Measure{T <: JuMP.AbstractJuMPScalar, V <: AbstractMeasureData}
    func::T
    data::V
end

"""
    InfiniteModel <: JuMP.AbstractModel

A DataType for storing all of the mathematical modeling information needed to
model an optmization problem with an infinite dimensional decision space.

**Fields**
- `next_meas_index::Int64` Index - 1 of next measure.
- `measures::Dict{Int64, Measure}` Measure indices to measure datatypes.
- `meas_to_name::Dict{Int64, String}` Measure indices to names.
- `meas_to_constrs::Dict{Int64, Vector{Int64}}` Measure indices to dependent
                                            constraint indices.
- `meas_to_meas::Dict{Int64, Vector{Int64}}` Measure indices to dependent
                                         measure indices.
- `meas_in_objective::Dict{Int64, Bool}` Measure indices to if used in objective.
- `next_param_index::Int64` Index - 1 of next infinite parameter.
- `next_param_id::Int64` Index - 1 of the next infinite parameter group.
- `params::Dict{Int64, InfOptParameter}` Infinite parameter indices to parameter
                                       datatype.
- `param_to_name::Dict{Int64, String}` Infinite parameter indices to names.
- `name_to_param::Union{Dict{String, Int64}, Nothing}` Names to infinite
                                                     parameters.
- `param_to_group_id::Dict{Int64, Int64}` Infinite parameter indices to group IDs.
- `param_to_constrs::Dict{Int64, Vector{Int64}}` Infinite parameter indices to list
                                             of dependent constraint indices.
- `param_to_meas::Dict{Int64, Vector{Int64}}` Infinite parameter indices to list
                                          of dependent measure indices.
- `param_to_vars::Dict{Int64, Vector{Int64}}` Infinite parameter indices to list
                                          of dependent variable indices.
- `next_var_index::Int64` Index - 1 of next variable index.
- `vars::Dict{Int64, Dict{Int64, Union{InfOptVariable, ReducedVariable}}` Variable
                                                  indices to variable datatype.
- `var_to_name::Dict{Int64, String}` Variable indices to names.
- `name_to_var::Union{Dict{String, Int64}, Nothing}` Variable names to indices.
- `var_to_lower_bound::Dict{Int64, Int64}` Variable indices to lower bound index.
- `var_to_upper_bound::Dict{Int64, Int64}` Variable indices to upper bound index.
- `var_to_fix::Dict{Int64, Int64}` Variable indices to fix index.
- `var_to_zero_one::Dict{Int64, Int64}` Variable indices to binary index.
- `var_to_integrality::Dict{Int64, Int64}` Variable indices to integer index.
- `var_to_constrs::Dict{Int64, Vector{Int64}}` Variable indices to dependent
                                           constraint indices.
- `var_to_meas::Dict{Int64, Vector{Int64}}` Variable indices to dependent
                                        measure indices.
- `var_in_objective::Dict{Int64, Bool}` Variable indices to if used in objective.
- `infinite_to_points::Dict{Int64, Vector{Int64}}` Infinite variable indices to
                                               dependent point variable indices.
- `infinite_to_reduced::Dict{Int64, Vector{Int64}}` Infinite variable indices to
                                               dependent reduced variable indices.
- `reduced_to_constrs::Dict{Int64, Vector{Int64}}` Reduced variable indices to dependent
                                               constraint indices.
- `reduced_to_meas::Dict{Int64, Vector{Int64}}` Reduced variable indices to dependent
                                            measure indices.
- `reduced_info::Dict{Int64, AbstractReducedInfo}` Reduced variable indices to
                                                 reduced variable information.
- `next_constr_index::Int64` Index - 1 of next constraint.
- `constrs::Dict{Int64, JuMP.AbstractConstraint}` Constraint indices to constraint
                                                datatypes.
- `constr_to_name::Dict{Int64, String}` Constraint indices to names.
- `name_to_constr::Union{Dict{String, Int64}, Nothing}` Constraint names to
                                                      indices.
- `constr_in_var_info::Dict{Int64, Bool}` Constraint indices to if related to
                                        variable information constraints.
- `objective_sense::MOI.OptimizationSense` Objective sense.
- `objective_function::JuMP.AbstractJuMPScalar` Finite scalar function.
- `obj_dict::Dict{Symbol, Any}` Store Julia symbols used with `InfiniteModel`
- `optimizer_factory::Union{JuMP.OptimizerFactory, Nothing}` Optimizer
                                                             information.
- `optimizer_model::JuMP.Model` Model used to solve `InfiniteModel`
- `ready_to_optimize::Bool` Is the optimizer_model up to date.
"""
mutable struct InfiniteModel <: JuMP.AbstractModel
    # Measure Data
    next_meas_index::Int64
    measures::Dict{Int64, Measure}
    meas_to_name::Dict{Int64, String}
    meas_to_constrs::Dict{Int64, Vector{Int64}}
    meas_to_meas::Dict{Int64, Vector{Int64}}
    meas_in_objective::Dict{Int64, Bool}

    # Parameter Data
    next_param_index::Int64
    next_param_id::Int64
    params::Dict{Int64, InfOptParameter}
    param_to_name::Dict{Int64, String}
    name_to_param::Union{Dict{String, Int64}, Nothing}
    param_to_group_id::Dict{Int64, Int64}
    param_to_constrs::Dict{Int64, Vector{Int64}}
    param_to_meas::Dict{Int64, Vector{Int64}}
    param_to_vars::Dict{Int64, Vector{Int64}}

    # Variable data
    next_var_index::Int64
    vars::Dict{Int64, InfOptVariable}
    var_to_name::Dict{Int64, String}
    name_to_var::Union{Dict{String, Int64}, Nothing}
    var_to_lower_bound::Dict{Int64, Int64}
    var_to_upper_bound::Dict{Int64, Int64}
    var_to_fix::Dict{Int64, Int64}
    var_to_zero_one::Dict{Int64, Int64}
    var_to_integrality::Dict{Int64, Int64}
    var_to_constrs::Dict{Int64, Vector{Int64}}
    var_to_meas::Dict{Int64, Vector{Int64}}
    var_in_objective::Dict{Int64, Bool}
    infinite_to_points::Dict{Int64, Vector{Int64}}
    infinite_to_reduced::Dict{Int64, Vector{Int64}}

    # Placeholder
    reduced_to_constrs::Dict{Int64, Vector{Int64}}
    reduced_to_meas::Dict{Int64, Vector{Int64}}
    reduced_info::Dict{Int64, AbstractReducedInfo}

    # Constraint Data
    next_constr_index::Int64
    constrs::Dict{Int64, JuMP.AbstractConstraint}
    constr_to_name::Dict{Int64, String}
    name_to_constr::Union{Dict{String, Int64}, Nothing}
    constr_in_var_info::Dict{Int64, Bool}

    # Objective Data
    objective_sense::MOI.OptimizationSense
    objective_function::JuMP.AbstractJuMPScalar

    # Objects
    obj_dict::Dict{Symbol, Any}

    # Optimize Data
    optimizer_factory::Union{JuMP.OptimizerFactory, Nothing}
    optimizer_model::JuMP.Model
    ready_to_optimize::Bool
end

"""
    InfiniteModel([optimizer_factory::JuMP.OptimizerFactory];
                  [caching_mode::MOIU.CachingOptimizerMode = MOIU.AUTOMATIC,
                  bridge_constraints::Bool = true])

Return a new infinite model where an optimizer is specified if an
`optimizer_factory` is given via [`JuMP.with_optimizer`](@ref). The optimizer
can also later be set with the [`JuMP.optimize!`](@ref) call. By default
the `optimizer_model` data field is initialized with a
[`TranscriptionModel`](@ref), but a different type of model can be assigned via
[`set_optimizer_model`](@ref) as can be required by extensions.

**Example**
```julia
julia> model = InfiniteModel()
An InfiniteOpt Model
Feasibility problem with:
Variables: 0
Optimizer model backend information:
Model mode: AUTOMATIC
CachingOptimizer state: NO_OPTIMIZER
Solver name: No optimizer attached.

julia> model = InfiniteModel(with_optimizer(Gurobi.Optimizer))
An InfiniteOpt Model
Feasibility problem with:
Variables: 0
Optimizer model backend information:
Model mode: AUTOMATIC
CachingOptimizer state: EMPTY_OPTIMIZER
Solver name: Gurobi
```
"""
function InfiniteModel(; kwargs...)
    return InfiniteModel(# Measures
                         0, Dict{Int64, Measure}(), Dict{Int64, String}(),
                         Dict{Int64, Vector{Int64}}(), Dict{Int64, Vector{Int64}}(),
                         Dict{Int64, Bool}(),
                         # Parameters
                         0, 0, Dict{Int64, InfOptParameter}(), Dict{Int64, String}(),
                         nothing, Dict{Int64, Int64}(), Dict{Int64, Vector{Int64}}(),
                         Dict{Int64, Vector{Int64}}(), Dict{Int64, Vector{Int64}}(),
                         # Variables
                         0, Dict{Int64, JuMP.AbstractVariable}(),
                         Dict{Int64, String}(), nothing, Dict{Int64, Int64}(),
                         Dict{Int64, Int64}(), Dict{Int64, Int64}(), Dict{Int64, Int64}(),
                         Dict{Int64, Int64}(), Dict{Int64, Vector{Int64}}(),
                         Dict{Int64, Vector{Int64}}(), Dict{Int64, Bool}(),
                         Dict{Int64, Vector{Int64}}(), Dict{Int64, Vector{Int64}}(),
                         # Placeholder variables
                         Dict{Int64, Vector{Int64}}(), Dict{Int64, Vector{Int64}}(),
                         Dict{Int64, Tuple}(),
                         # Constraints
                         0, Dict{Int64, JuMP.AbstractConstraint}(),
                         Dict{Int64, String}(), nothing, Dict{Int64, Bool}(),
                         # Objective
                         MOI.FEASIBILITY_SENSE,
                         zero(JuMP.GenericAffExpr{Float64, InfOptVariableRef}),
                         # Object dictionary
                         Dict{Symbol, Any}(),
                         # Optimize data
                         nothing, TranscriptionModel(;kwargs...), false)
end

# Dispatch for InfiniteModel call with optimizer factory
function InfiniteModel(optimizer_factory::JuMP.OptimizerFactory; kwargs...)
    model = InfiniteModel()
    model.optimizer_factory = optimizer_factory
    model.optimizer_model = TranscriptionModel(optimizer_factory; kwargs...)
    return model
end

# Define basic InfiniteModel extensions
Base.broadcastable(model::InfiniteModel) = Ref(model)
JuMP.object_dictionary(model::InfiniteModel) = model.obj_dict

"""
    InfOptVariableRef <: JuMP.AbstractVariableRef
A DataType for untranscripted references of all types (parameters, variables,
measures, etc.).

**Fields**
- `model::InfiniteModel` Infinite model.
- `index::Int64` Index of variable in model.
- `type::Symbol` Type of variable
"""
struct InfOptVariableRef <: JuMP.AbstractVariableRef
    model::InfiniteModel
    index::Int64
    type::Symbol
end

# function that accesses the type of variable under an InfOptVariableRef
function JuMP.variable_type(vref::InfOptVariableRef)::Symbol
    return vref.type
end

"""
    InfiniteVariable{S, T, U, V} <: InfOptVariable
A DataType for storing core infinite variable information. Note each element of
the parameter reference tuple must contain either a single
[`ParameterRef`](@ref) or an `AbstractArray` of `ParameterRef`s where each
`ParameterRef` has the same group ID number.

**Fields**
- `info::JuMP.VariableInfo{S, T, U, V}` JuMP variable information.
- `parameter_refs::Tuple` The infinite parameters(s) that parameterize the
                          variable.
"""
struct InfiniteVariable{S, T, U, V} <: InfOptVariable
    info::JuMP.VariableInfo{S, T, U, V}
    parameter_refs::Tuple
end

"""
    PointVariable{S, T, U, V} <: InfOptVariable
A DataType for storing point variable information. Note that the elements
`parameter_values` field must match the format of the parameter reference tuple
defined in [`InfiniteVariable`](@ref)

**Fields**
- `info::JuMP.VariableInfo{S, T, U, V}` JuMP Variable information.
- `infinite_variable_ref::InfOptVariableRef` The infinite variable associated
                                               with the point variable.
- `parameter_values::Tuple` The infinite parameter values defining the point.
"""
struct PointVariable{S, T, U, V} <: InfOptVariable
    info::JuMP.VariableInfo{S, T, U, V}
    infinite_variable_ref::InfOptVariableRef
    parameter_values::Tuple
end

"""
    GlobalVariable{S, T, U, V} <: InfOptVariable
A DataType for storing global variable information.

**Fields**
- `info::JuMP.VariableInfo{S, T, U, V}` JuMP variable information.
"""
struct GlobalVariable{S, T, U, V} <: InfOptVariable
    info::JuMP.VariableInfo{S, T, U, V}
end

"""
    ReducedInfiniteInfo <: AbstractReducedInfo

A DataType for storing reduced infinite variable information.

**Fields**
- `infinite_variable_ref::InfiniteVariableRef` The original infinite variable.
- `eval_supports::Dict{Int64, Union{Number, JuMP.Containers.SparseAxisArray{<:Number}}}`
  The original parameter tuple indices to the evaluation supports.
"""
struct ReducedInfiniteInfo <: AbstractReducedInfo
#    infinite_variable_ref::InfiniteVariableRef
    infinite_variable_ref::InfOptVariableRef
    eval_supports::Dict{Int64, Union{Number,
                                   JuMPC.SparseAxisArray{<:Number}}}
end

"""
    DiscreteMeasureData <: AbstractMeasureData

A DataType for one dimensional measure abstraction data where the measure
abstraction is of the form:
``measure = \\int_{\\tau \\in T} f(\\tau) w(\\tau) d\\tau \\approx \\sum_{i = 1}^N \\alpha_i f(\\tau_i) w(\\tau_i)``.

**Fields**
- `parameter_ref::InfOptVariableRef` The infinite parameter over which the
                                integration occurs.
- `coefficients::Vector{<:Number}` Coefficients ``\\alpha_i`` for the above
                                   measure abstraction.
- `supports::Vector{<:Number}` Support points ``\\tau_i`` for the above
                               measure abstraction.
- `name::String` Name of the measure that will be implemented.
- `weight_function::Function` Weighting function ``w`` must map support value
                              input value of type `Number` to a scalar value.
"""
struct DiscreteMeasureData <: AbstractMeasureData
    parameter_ref::InfOptVariableRef
    coefficients::Vector{<:Number}
    supports::Vector{<:Number}
    name::String
    weight_function::Function
    function DiscreteMeasureData(parameter_ref::InfOptVariableRef,
                                 coeffs::Vector{<:Number},
                                 supports::Vector{<:Number},
                                 name::String, weight_func::Function)
        if length(coeffs) != length(supports)
            error("The amount of coefficients must match the amount of " *
                  "support points.")
        end
        if JuMP.has_lower_bound(parameter_ref)
            check1 = minimum(supports) < JuMP.lower_bound(parameter_ref)
            check2 = maximum(supports) > JuMP.upper_bound(parameter_ref)
            if check1 || check2
                error("Support points violate parameter bounds.")
            end
        end
        return new(parameter_ref, coeffs, supports, name, weight_func)
    end
end

"""
    MultiDiscreteMeasureData <: AbstractMeasureData

A DataType for multi-dimensional measure abstraction data where the measure
abstraction is of the form:
``measure = \\int_{\\tau \\in T} f(\\tau) w(\\tau) d\\tau \\approx \\sum_{i = 1}^N \\alpha_i f(\\tau_i) w(\\tau_i)``.

**Fields**
- `parameter_ref::JuMP.Containers.SparseAxisArray{<:InfOptVariableRef}` The infinite
   parameters over which the integration occurs.
- `coefficients::Vector{<:Number}` Coefficients ``\\alpha_i`` for the above
                                   measure abstraction.
- `supports::Vector{<:JuMP.Containers.SparseAxisArray{<:Number}}` Support points
   ``\\tau_i`` for the above measure abstraction.
- `name::String` Name of the measure that will be implemented.
- `weight_function::Function` Weighting function ``w`` must map a numerical
                              support of type `JuMP.Containers.SparseAxisArray`
                              to a scalar value.
"""
struct MultiDiscreteMeasureData <: AbstractMeasureData
    parameter_ref::JuMPC.SparseAxisArray{<:InfOptVariableRef}
    coefficients::Vector{<:Number}
    supports::Vector{<:JuMPC.SparseAxisArray}
    name::String
    weight_function::Function
    function MultiDiscreteMeasureData(parameter_ref::JuMPC.SparseAxisArray{<:InfOptVariableRef},
                                      coeffs::Vector{<:Number},
                                      supports::Vector{<:JuMPC.SparseAxisArray},
                                      name::String, weight_func::Function)
        if length(coeffs) != length(supports)
            error("The amount of coefficients must match the amount of " *
                  "support points.")
        elseif !_only_one_group(parameter_ref)
            error("Parameters must all have same group ID.")
        elseif length(supports) == 0
            error("Must specify at least one support.")
        elseif keys(supports[1].data) != keys(parameter_ref.data)
            error("The keys/dimensions of the support points and parameters " *
                  "do not match.")
        end
        for i in eachindex(supports)
            for key in keys(parameter_ref.data)
                support = supports[i].data[key]
                pref = parameter_ref.data[key]
                if JuMP.has_lower_bound(pref)
                    check1 = support < JuMP.lower_bound(pref)
                    check2 = support > JuMP.upper_bound(pref)
                    if check1 || check2
                        error("Support points violate parameter bounds.")
                    end
                end
            end
        end
        return new(parameter_ref, coeffs, supports, name, weight_func)
    end
end

"""
    IntervalSet <: AbstractInfiniteSet

A DataType that stores the lower and upper interval bounds for infinite
parameters that are continuous over a certain that interval.

**Fields**
- `lower_bound::Float64` Lower bound of the infinite parameter.
- `upper_bound::Float64` Upper bound of the infinite parameter.
"""
struct IntervalSet <: AbstractInfiniteSet
    lower_bound::Float64
    upper_bound::Float64
end

"""
    IntervalSet(lower_bound::Number, upper_bound::Number)

A constructor for [`IntervalSet`](@ref) that converts values of type `Number` to
values of type `Float64` as required by `IntervalSet`.
"""
IntervalSet(lb::Number, ub::Number) = IntervalSet(convert(Float64, lb),
                                                  convert(Float64, ub))

"""
    DistributionSet{T <: Distributions.NonMatrixDistribution} <: AbstractInfiniteSet

A DataType that stores the distribution characterizing infinite parameters that
are random.

**Fields**
- `distribution::T` Distribution of the random parameter.
"""
struct DistributionSet{T <: Distributions.NonMatrixDistribution} <: AbstractInfiniteSet
    distribution::T
end

"""
    BoundedScalarConstraint{F <: JuMP.AbstractJuMPScalar,
                            S <: MOI.AbstractScalarSet} <: JuMP.AbstractConstraint

A DataType that stores infinite constraints defined on a subset of the infinite
parameters on which they depend.

**Fields**
- `func::F` The JuMP object.
- `set::S` The MOI set.
- `bounds::Dict{InfOptVariableRef, IntervalSet}` A dictionary mapping parameter
                                                 references to an interval set.
"""
struct BoundedScalarConstraint{F <: JuMP.AbstractJuMPScalar,
                               S <: MOI.AbstractScalarSet} <: JuMP.AbstractConstraint
    func::F
    set::S
    bounds::Dict{InfOptVariableRef, IntervalSet}
end

"""
InfOptConstraintRef{S <: JuMP.AbstractShape}

A DataType for constraint references unique to InfiniteOpt. Types could be
Infinite, Measure, or Finite
**Fields**
- `model::InfiniteModel` Infinite model.
- `index::Int64` Index of constraint in model.
- `shape::JuMP.AbstractShape` Shape of constraint
- `type::Symbol`
"""

struct InfOptConstraintRef{S <: JuMP.AbstractShape}
    model::InfiniteModel
    index::Int64
    shape::S
    type::Symbol
end

# function that accesses the type of variable under a InfOptConstraintRef
function constraint_type(cref::InfOptConstraintRef)::Symbol
    return cref.type
end

# Helper functions that check the type of an expression
function all_types_of_expr(expr::InfOptVariableRef)::Array{Symbol, 1}
    return [variable_type(expr)]
end

function all_types_of_expr(expr::JuMP.GenericAffExpr)::Array{Symbol, 1}
    return unique(variable_type.(expr.terms.keys))
end

function all_types_of_expr(expr::JuMP.GenericQuadExpr)::Array{Symbol, 1}
    types = all_types_of_expr(expr.aff)
    types = cat(types, [variable_type(i.a) for i in expr.terms.keys],
                       [variable_type(i.b) for i in expr.terms.keys], dims = 1)
    return unique(types)
end

function all_types_of_expr(expr::JuMP.VariableRef)::Array{Symbol, 1}
    return [VariableRef]
end


function _is_single_type_expr(expr::JuMP.AbstractJuMPScalar, type::Symbol)::Bool
    if expr isa InfOptVariableRef
        return variable_type(expr) == type
    elseif expr isa JuMP.GenericAffExpr{Float64, InfOptVariableRef} ||
           expr isa JuMP.GenericQuadExpr{Float64, InfOptVariableRef}
        types_of_expr = all_types_of_expr(expr)
        return length(types_of_expr) == 1 && type in types_of_expr
    else
        return false
    end
end

function _is_infinite_expr(expr::JuMP.AbstractJuMPScalar)::Bool
    if expr isa InfOptVariableRef
        return variable_type(expr) == Infinite || variable_type(expr) == Reduced
    else
        return expr isa JuMP.GenericAffExpr{Float64, InfOptVariableRef} ||
               expr isa JuMP.GenericQuadExpr{Float64, InfOptVariableRef}
    end
    #return false
end

function _is_measure_expr(expr::JuMP.AbstractJuMPScalar)::Bool
    if expr isa InfOptVariableRef
        return variable_type(expr) == MeasureRef
    elseif expr isa JuMP.GenericAffExpr{Float64, InfOptVariableRef} ||
           expr isa JuMP.GenericQuadExpr{Float64, InfOptVariableRef}
        types_of_expr = all_types_of_expr(expr)
        if Infinite in types_of_expr || Reduced in types_of_expr ||
                                        Parameter in types_of_expr
            return false
        elseif MeasureRef in types_of_expr
            return true
        end
#        return MeasureRef in types_of_expr
    end
    return false
end

function _is_finite_expr(expr::JuMP.AbstractJuMPScalar)::Bool
    if expr isa InfOptVariableRef
        return variable_type(expr) == Point || variable_type(expr) == Global
    elseif expr isa JuMP.GenericAffExpr{Float64, InfOptVariableRef} ||
           expr isa JuMP.GenericQuadExpr{Float64, InfOptVariableRef}
        types_of_expr = all_types_of_expr(expr)
        if Infinite in types_of_expr || Reduced in types_of_expr ||
           Parameter in types_of_expr || MeasureRef in types_of_expr
            return false
        else
            return true
        end
    end
    return false
end

function _is_parameter_expr(expr::JuMP.AbstractJuMPScalar)::Bool
    if expr isa InfOptVariableRef
        return variable_type(expr) == Parameter
    elseif expr isa JuMP.GenericAffExpr{Float64, InfOptVariableRef} ||
           expr isa JuMP.GenericQuadExpr{Float64, InfOptVariableRef}
        types_of_expr = all_types_of_expr(expr)
        return types_of_expr == [Parameter]
#        return Parameter in types_of_expr && !(Infinite in types_of_expr) &&
#               !(Reduced in types_of_expr) && !(Measure in types_of_expr)
    end
    return false
end

function _expr_constraint_type(expr::JuMP.AbstractJuMPScalar)::Symbol
    if _is_finite_expr(expr)
        return Finite
    elseif _is_measure_expr(expr)
        return MeasureRef
    elseif _is_infinite_expr(expr)
        return Infinite
    else
        error("expr is not of any valid InfiniteOpt constraint type.")
    end
end
