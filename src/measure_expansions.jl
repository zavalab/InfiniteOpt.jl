# Helper function to get reduced variable info
function _reduced_info(vref::ReducedInfiniteVariableRef)::ReducedInfiniteInfo
    return JuMP.owner_model(vref).reduced_info[JuMP.index(vref)]
end

"""
    infinite_variable_ref(vref::ReducedInfiniteVariableRef)::InfiniteVariableRef

Return the `InfiniteVariableRef` associated with the reduced infinite variable
`vref`.

**Example**
```julia
julia> infinite_variable_ref(vref)
g(t, x)
```
"""
function infinite_variable_ref(vref::ReducedInfiniteVariableRef)::InfiniteVariableRef
    return _reduced_info(vref).infinite_variable_ref
end

"""
    eval_supports(vref::ReducedInfiniteVariableRef)::Dict

Return the evaluation supports associated with the reduced infinite variable
`vref`.

**Example**
```julia
julia> eval_supports(vref)
Dict{Int64,Float64} with 1 entry:
  1 => 0.5
```
"""
function eval_supports(vref::ReducedInfiniteVariableRef)::Dict
    return _reduced_info(vref).eval_supports
end

"""
    parameter_refs(vref::ReducedInfiniteVariableRef)::Tuple

Return the `ParameterRef`(s) associated with the reduced infinite variable
`vref`. This is formatted as a Tuple of containing the parameter references as
they were inputted to define the untracripted infinite variable except, the
evaluated parameters are excluded.

**Example**
```julia
julia> parameter_refs(vref)
(t,   [2]  =  x[2]
  [1]  =  x[1])
```
"""
function parameter_refs(vref::ReducedInfiniteVariableRef)
    orig_prefs = parameter_refs(infinite_variable_ref(vref))
    prefs = Tuple(orig_prefs[i] for i = 1:length(orig_prefs) if !haskey(eval_supports(vref), i))
    return prefs
end

"""
    JuMP.name(vref::ReducedInfiniteVariableRef)::String

Extend `JuMP.name` to return name of reduced infinite variable references. This
is used when displaying measure expansions that contain such variables.

**Exanple**
```julia
julia> name(rvref)
g(1.25, x)
```
"""
function JuMP.name(vref::ReducedInfiniteVariableRef)::String
    root_name = _root_name(infinite_variable_ref(vref))
    prefs = parameter_refs(infinite_variable_ref(vref))
    param_names = _root_names(prefs)
    for (k, v) in eval_supports(vref)
        param_names[k] = string(v)
    end
    param_name_tuple = "("
    for i = 1:length(param_names)
        if i != length(param_names)
            param_name_tuple *= string(param_names[i], ", ")
        else
            param_name_tuple *= string(param_names[i])
        end
    end
    param_name_tuple *= ")"
    return string(root_name, param_name_tuple)
end

"""
    JuMP.has_lower_bound(vref::ReducedInfiniteVariableRef)::Bool

Extend [`JuMP.has_lower_bound`](@ref) to return a `Bool` whether the original
infinite variable of `vref` has a lower bound.

**Example**
```julia
julia> has_lower_bound(vref)
true
```
"""
function JuMP.has_lower_bound(vref::ReducedInfiniteVariableRef)::Bool
    return JuMP.has_lower_bound(infinite_variable_ref(vref))
end

"""
    JuMP.lower_bound(vref::ReducedInfiniteVariableRef::Float64

Extend [`JuMP.lower_bound`](@ref) to return the lower bound of the original
infinite variable of `vref`. Errors if `vref` doesn't have a lower bound.

**Example**
```julia
julia> lower_bound(vref)
0.0
```
"""
function JuMP.lower_bound(vref::ReducedInfiniteVariableRef)::Float64
    if !JuMP.has_lower_bound(vref)
        error("Variable $(vref) does not have a lower bound.")
    end
    return JuMP.lower_bound(infinite_variable_ref(vref))
end

"""
    JuMP.lower_bound_index(vref::ReducedInfiniteVariableRef)::Int

Extend [`JuMP.lower_bound_index`](@ref) to return the index of the lower bound
constraint associated with the original infinite variable of `vref`. Errors if
`vref` does not have a lower bound.

**Example**
```julia
julia> lower_bound_index(vref)
2
```
"""
function JuMP.lower_bound_index(vref::ReducedInfiniteVariableRef)::Int
    if !JuMP.has_lower_bound(vref)
        error("Variable $(vref) does not have a lower bound.")
    end
    return JuMP.lower_bound_index(infinite_variable_ref(vref))
end

"""
    JuMP.LowerBoundRef(vref::ReducedInfiniteVariableRef)::GeneralConstraintRef

Extend [`JuMP.LowerBoundRef`](@ref) to extract a constraint reference for the
lower bound of the original infinite variable of `vref`.

**Example**
```julia
julia> cref = LowerBoundRef(vref)
var >= 0.0
```
"""
function JuMP.LowerBoundRef(vref::ReducedInfiniteVariableRef)::GeneralConstraintRef
    return JuMP.LowerBoundRef(infinite_variable_ref(vref))
end

"""
    JuMP.has_upper_bound(vref::ReducedInfiniteVariableRef)::Bool

Extend [`JuMP.has_upper_bound`](@ref) to return a `Bool` whether the original
infinite variable of `vref` has an upper bound.

**Example**
```julia
julia> has_upper_bound(vref)
true
```
"""
function JuMP.has_upper_bound(vref::ReducedInfiniteVariableRef)::Bool
    return JuMP.has_upper_bound(infinite_variable_ref(vref))
end

"""
    JuMP.upper_bound(vref::ReducedInfiniteVariableRef)::Float64

Extend [`JuMP.upper_bound`](@ref) to return the upper bound of the original
infinite variable of `vref`. Errors if `vref` doesn't have a upper bound.

**Example**
```julia
julia> upper_bound(vref)
0.0
```
"""
function JuMP.upper_bound(vref::ReducedInfiniteVariableRef)::Float64
    if !JuMP.has_upper_bound(vref)
        error("Variable $(vref) does not have a upper bound.")
    end
    return JuMP.upper_bound(infinite_variable_ref(vref))
end

"""
    JuMP.upper_bound_index(vref::ReducedInfiniteVariableRef)::Int

Extend [`JuMP.upper_bound_index`](@ref) to return the index of the upper bound
constraint associated with the original infinite variable of `vref`. Errors if
`vref` does not have a upper bound.

**Example**
```julia
julia> upper_bound_index(vref)
2
```
"""
function JuMP.upper_bound_index(vref::ReducedInfiniteVariableRef)::Int
    if !JuMP.has_upper_bound(vref)
        error("Variable $(vref) does not have a upper bound.")
    end
    return JuMP.upper_bound_index(infinite_variable_ref(vref))
end

"""
    JuMP.UpperBoundRef(vref::ReducedInfiniteVariableRef)::GeneralConstraintRef

Extend [`JuMP.UpperBoundRef`](@ref) to extract a constraint reference for the
upper bound of the original infinite variable of `vref`.

**Example**
```julia
julia> cref = UpperBoundRef(vref)
var <= 1.0
```
"""
function JuMP.UpperBoundRef(vref::ReducedInfiniteVariableRef)::GeneralConstraintRef
    return JuMP.UpperBoundRef(infinite_variable_ref(vref))
end

"""
    JuMP.is_fixed(vref::ReducedInfiniteVariableRef)::Bool

Extend [`JuMP.is_fixed`](@ref) to return `Bool` whether the original infinite
variable of `vref` is fixed.

**Example**
```julia
julia> is_fixed(vref)
true
```
"""
function JuMP.is_fixed(vref::ReducedInfiniteVariableRef)::Bool
    return JuMP.is_fixed(infinite_variable_ref(vref))
end

"""
    JuMP.fix_value(vref::ReducedInfiniteVariableRef)::Float64

Extend [`JuMP.fix_value`](@ref) to return the fix value of the original infinite
variable of `vref`. Errors if variable is not fixed.

**Example**
```julia
julia> fix_value(vref)
0.0
```
"""
function JuMP.fix_value(vref::ReducedInfiniteVariableRef)::Float64
    if !JuMP.is_fixed(vref)
        error("Variable $(vref) is not fixed.")
    end
    return JuMP.fix_value(infinite_variable_ref(vref))
end

"""
    JuMP.fix_index(vref::ReducedInfiniteVariableRef)::Int

Extend [`JuMP.fix_index`](@ref) to return the index of the fix constraint
associated with the original infinite variable of `vref`. Errors if `vref` is
not fixed.

**Example**
```julia
julia> fix_index(vref)
2
```
"""
function JuMP.fix_index(vref::ReducedInfiniteVariableRef)::Int
    if !JuMP.is_fixed(vref)
        error("Variable $(vref) is not fixed.")
    end
    return JuMP.fix_index(infinite_variable_ref(vref))
end

"""
    JuMP.FixRef(vref::ReducedInfiniteVariableRef)::GeneralConstraintRef

Extend [`JuMP.FixRef`](@ref) to return the constraint reference of the fix
constraint associated with the original infinite variable of `vref`. Errors
`vref` is not fixed.

**Examples**
```julia
julia> cref = FixRef(vref)
var == 1.0
```
"""
function JuMP.FixRef(vref::ReducedInfiniteVariableRef)::GeneralConstraintRef
    return JuMP.FixRef(infinite_variable_ref(vref))
end

"""
    JuMP.start_value(vref::ReducedInfiniteVariableRef)::Union{Nothing, Float64}

Extend [`JuMP.start_value`](@ref) to return starting value of the original
infinite variable of `vref` if it has one. Returns `nothing` otherwise.

**Example**
```julia
julia> start_value(vref)
0.0
```
"""
function JuMP.start_value(vref::ReducedInfiniteVariableRef)::Union{Nothing, Float64}
    return JuMP.start_value(infinite_variable_ref(vref))
end

"""
    JuMP.is_binary(vref::ReducedInfiniteVariableRef)::Bool

Extend [`JuMP.is_binary`](@ref) to return `Bool` whether the original infinite
variable of `vref` is binary.

**Example**
```julia
julia> is_binary(vref)
true
```
"""
function JuMP.is_binary(vref::ReducedInfiniteVariableRef)::Bool
    return JuMP.is_binary(infinite_variable_ref(vref))
end

"""
    JuMP.binary_index(vref::ReducedInfiniteVariableRef)::Int

Extend [`JuMP.binary_index`](@ref) to return the index of the binary constraint
associated with the original infinite variable of `vref`. Errors if `vref` is
not binary.

**Example**
```julia
julia> binary_index(vref)
2
```
"""
function JuMP.binary_index(vref::ReducedInfiniteVariableRef)::Int
    if !JuMP.is_binary(vref)
        error("Variable $(vref) is not binary.")
    end
    return JuMP.binary_index(infinite_variable_ref(vref))
end

"""
    JuMP.BinaryRef(vref::ReducedInfiniteVariableRef)::GeneralConstraintRef

Extend [`JuMP.BinaryRef`](@ref) to return a constraint reference to the
constraint constrainting the original infinite variable of `vref` to be binary.
Errors if one does not exist.

**Example**
```julia
julia> cref = BinaryRef(vref)
var binary
```
"""
function JuMP.BinaryRef(vref::ReducedInfiniteVariableRef)::GeneralConstraintRef
    return JuMP.BinaryRef(infinite_variable_ref(vref))
end

"""
    JuMP.is_integer(vref::ReducedInfiniteVariableRef)::Bool

Extend [`JuMP.is_integer`](@ref) to return `Bool` whether the original infinite
variable of `vref` is integer.

**Example**
```julia
julia> is_integer(vref)
true
```
"""
function JuMP.is_integer(vref::ReducedInfiniteVariableRef)::Bool
    return JuMP.is_integer(infinite_variable_ref(vref))
end

"""
    JuMP.integer_index(vref::ReducedInfiniteVariableRef)::Int

Extend [`JuMP.integer_index`](@ref) to return the index of the integer constraint
associated with the original infinite variable of `vref`. Errors if `vref` is
not integer.

**Example**
```julia
julia> integer_index(vref)
2
```
"""
function JuMP.integer_index(vref::ReducedInfiniteVariableRef)::Int
    if !JuMP.is_integer(vref)
        error("Variable $(vref) is not an integer.")
    end
    return JuMP.integer_index(infinite_variable_ref(vref))
end

"""
    JuMP.IntegerRef(vref::ReducedInfiniteVariableRef)::GeneralConstraintRef

Extend [`JuMP.IntegerRef`](@ref) to return a constraint reference to the
constraint constrainting the original infinite variable of `vref` to be integer.
Errors if one does not exist.

**Example**
```julia
julia> cref = IntegerRef(vref)
var integer
```
"""
function JuMP.IntegerRef(vref::ReducedInfiniteVariableRef)::GeneralConstraintRef
    return JuMP.IntegerRef(infinite_variable_ref(vref))
end

"""
    used_by_constraint(vref::ReducedInfiniteVariableRef)::Bool

Return a `Bool` indicating if `vref` is used by a constraint.

**Example**
```julia
julia> used_by_constraint(vref)
false
```
"""
function used_by_constraint(vref::ReducedInfiniteVariableRef)::Bool
    return haskey(JuMP.owner_model(vref).reduced_to_constrs, JuMP.index(vref))
end

"""
    used_by_measure(vref::ReducedInfiniteVariableRef)::Bool

Return a `Bool` indicating if `vref` is used by a measure.

**Example**
```julia
julia> used_by_measure(vref)
true
```
"""
function used_by_measure(vref::ReducedInfiniteVariableRef)::Bool
    return haskey(JuMP.owner_model(vref).reduced_to_meas, JuMP.index(vref))
end

"""
    JuMP.is_valid(model::InfiniteModel, vref::ReducedInfiniteVariableRef)::Bool

Extend [`JuMP.is_valid`](@ref) to accomodate reduced infinite variables.

**Example**
```julia
julia> is_valid(model, vref)
true
```
"""
function JuMP.is_valid(model::InfiniteModel,
                       vref::ReducedInfiniteVariableRef)::Bool
    return (model === JuMP.owner_model(vref) && JuMP.index(vref) in keys(model.reduced_info))
end

"""
    JuMP.delete(model::InfiniteModel, vref::ReducedInfiniteVariableRef)
Extend the `JuMP.delete` function to accomodate our new variable types.
"""
function JuMP.delete(model::InfiniteModel, vref::ReducedInfiniteVariableRef)
    # check valid reference
    @assert JuMP.is_valid(model, vref) "Invalid variable reference."
    # remove from measures if used
    if used_by_measure(vref)
        for mindex in model.reduced_to_meas[JuMP.index(vref)]
            if isa(model.measures[mindex].func, ReducedInfiniteVariableRef)
                data = model.measures[mindex].data
                model.measures[mindex] = Measure(zero(JuMP.AffExpr), data)
            else
                _remove_variable(model.measures[mindex].func, vref)
            end
            mref = MeasureRef(model, mindex)
            JuMP.set_name(mref, _make_meas_name(model.measures[mindex]))
        end
        delete!(model.reduced_to_meas, JuMP.index(vref))
    end
    # remove from constraints if used
    if used_by_constraint(vref)
        for cindex in model.reduced_to_constrs[JuMP.index(vref)]
            if isa(model.constrs[cindex].func, ReducedInfiniteVariableRef)
                set = model.constrs[cindex].set
                model.constrs[cindex] = JuMP.ScalarConstraint(zero(JuMP.AffExpr),
                                                                   set)
            else
                _remove_variable(model.constrs[cindex].func, vref)
            end
        end
        delete!(model.reduced_to_constrs, JuMP.index(vref))
    end
    # delete the info
    delete!(model.reduced_info, JuMP.index(vref))
    return
end

# Helper function for making place holder point variables
function _make_point_variable(ivref::InfiniteVariableRef)::PointVariableRef
    inf_model = JuMP.owner_model(ivref)
    index = inf_model.next_var_index += 1
    return PointVariableRef(inf_model, index)
end

## Helper function for making place holder infinite variables
# first time reduction
function _make_reduced_variable(ivref::InfiniteVariableRef, removed_index::Int,
                                support::Union{Number,
                                JuMP.Containers.SparseAxisArray{<:Number}}
                                )::ReducedInfiniteVariableRef
    inf_model = JuMP.owner_model(ivref)
    index = inf_model.next_var_index += 1
    inf_model.reduced_info[index] = ReducedInfiniteInfo(ivref,
                                                 Dict(removed_index => support))
    return ReducedInfiniteVariableRef(inf_model, index)
end

# further reduce
function _make_reduced_variable(ivref::InfiniteVariableRef,
                                supports::Dict)::ReducedInfiniteVariableRef
    inf_model = JuMP.owner_model(ivref)
    index = inf_model.next_var_index += 1
    inf_model.reduced_info[index] = ReducedInfiniteInfo(ivref, copy(supports))
    return ReducedInfiniteVariableRef(inf_model, index)
end

## Make helper functions for extracting evaluated parameter values
# DiscreteMeasureData
function _get_param_value_list(pref::ParameterRef, data::DiscreteMeasureData)
    return data.supports
end

# MultiDiscreteMeasureData
function _get_param_value_list(pref::ParameterRef,
                               data::MultiDiscreteMeasureData)
    key = first(filter(p -> p[2] == pref, data.parameter_ref.data))[1]
    return [data.supports[i][key] for i = 1:length(data.supports)]
end

## Implement functions for expanding measures into regular expressions
# InfiniteVariableRef
function _expand_measure(ivref::InfiniteVariableRef,
                         data::Union{DiscreteMeasureData,
                                     MultiDiscreteMeasureData},
                         trans_model::JuMP.AbstractModel,
                         point_mapper::Function)::JuMP.AbstractJuMPScalar
    # figure out the parameter groups
    group = group_id(first(data.parameter_ref))
    groups = _groups(parameter_refs(ivref))
    # prepare return AffExpr and get necessary information
    aff = zero(JuMP.GenericAffExpr{Float64, GeneralVariableRef})
    # treat variable as constant if doesn't have measure parameter
    if !(group in groups)
        for i = 1:length(data.supports)
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]),
                                    ivref)
        end
    # convert variable into point variables if its only parameter is the
    # measure parameter
    elseif length(parameter_refs(ivref)) == 1
        for i = 1:length(data.supports)
            pvref = _make_point_variable(ivref)
            point_mapper(trans_model, pvref, ivref, (data.supports[i],))
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]),
                                    pvref)
        end
    # make reduced variables if the variable contains other parameters
    else
        tuple_loc = findfirst(isequal(group), groups)
        for i = 1:length(data.supports)
            rvref = _make_reduced_variable(ivref, tuple_loc, data.supports[i])
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]),
                                    rvref)
        end
    end
    return aff
end

# ReducedInfiniteVariableRef
function _expand_measure(rvref::ReducedInfiniteVariableRef,
                         data::Union{DiscreteMeasureData,
                                     MultiDiscreteMeasureData},
                         trans_model::JuMP.AbstractModel,
                         point_mapper::Function)::JuMP.AbstractJuMPScalar
    # figure out the parameters used by the reduced infinite variable
    orig_prefs = parameter_refs(infinite_variable_ref(rvref))
    # figure out the parameter groups
    group = group_id(first(data.parameter_ref))
    groups = _groups(parameter_refs(rvref))
    # prepare return AffExpr and get necessary information
    aff = zero(JuMP.GenericAffExpr{Float64, GeneralVariableRef})
    # treat variable as constant if doesn't have measure parameter
    if !(group in groups)
        for i = 1:length(data.supports)
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]),
                                    rvref)
        end
    # convert variable into point variables if its only parameter is the
    # measure parameter
    elseif length(parameter_refs(rvref)) == 1
        tuple_loc = findfirst(isequal(group), _groups(orig_prefs))
        for i = 1:length(data.supports)
            pvref = _make_point_variable(infinite_variable_ref(rvref))
            _reduced_info(rvref).eval_supports[tuple_loc] = data.supports[i]
            support = Tuple(eval_supports(rvref)[j] for j = 1:length(eval_supports(rvref)))
            point_mapper(trans_model, pvref, infinite_variable_ref(rvref), support)
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]),
                                    pvref)
        end
    # make reduced variables if the variable contains other parameters
    else
        tuple_loc = findfirst(isequal(group), _groups(orig_prefs))
        for i = 1:length(data.supports)
            new_rvref = _make_reduced_variable(infinite_variable_ref(rvref),
                                               eval_supports(rvref))
            _reduced_info(new_rvref).eval_supports[tuple_loc] = data.supports[i]
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]),
                                    new_rvref)
        end
    end
    return aff
end

# FiniteVariableRef
function _expand_measure(vref::FiniteVariableRef,
                         data::Union{DiscreteMeasureData,
                                     MultiDiscreteMeasureData},
                         trans_model::JuMP.AbstractModel,
                         point_mapper::Function)::JuMP.AbstractJuMPScalar
    aff = zero(JuMP.GenericAffExpr{Float64, GeneralVariableRef})
    # treat the variable as a constant
    for i = 1:length(data.supports)
        JuMP.add_to_expression!(aff, data.coefficients[i] *
                                data.weight_function(data.supports[i]),
                                vref)
    end
    return aff
end

# ParameterRef with scalar data
function _expand_measure(pref::ParameterRef,
                         data::DiscreteMeasureData,
                         trans_model::JuMP.AbstractModel,
                         point_mapper::Function)::JuMP.AbstractJuMPScalar
    aff = zero(JuMP.GenericAffExpr{Float64, GeneralVariableRef})
    # replace the parameter with its value if it is the measure parameter
    if data.parameter_ref == pref
        for i = 1:length(data.supports)
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]) *
                                    data.supports[i])
        end
    # treat the parameter as a constant otherwise
    else
        for i = 1:length(data.supports)
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]),
                                    pref)
        end
    end
    return aff
end

# ParameterRef with vector data
function _expand_measure(pref::ParameterRef,
                         data::MultiDiscreteMeasureData,
                         trans_model::JuMP.AbstractModel,
                         point_mapper::Function)::JuMP.AbstractJuMPScalar
    aff = zero(JuMP.GenericAffExpr{Float64, GeneralVariableRef})
    # determine if pref is part of the measure parameters
    pref_dict = filter(p -> p[2] == pref, data.parameter_ref.data)
    # replace the parameter with its value if it is the measure parameter
    if length(pref_dict) != 0
        for i = 1:length(data.supports)
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]) *
                                    data.supports[i][collect(keys(pref_dict))[1]])
        end
    # treat the parameter as a constant otherwise
    else
        for i = 1:length(data.supports)
            JuMP.add_to_expression!(aff, data.coefficients[i] *
                                    data.weight_function(data.supports[i]),
                                    pref)
        end
    end
    return aff
end

# GenericAffExpr
function _expand_measure(expr::JuMP.GenericAffExpr,
                         data::Union{DiscreteMeasureData,
                                     MultiDiscreteMeasureData},
                         trans_model::JuMP.AbstractModel,
                         point_mapper::Function)::JuMP.AbstractJuMPScalar
    # need to use a quadratic expression in case contains measures with
    # quadratic expressions
    quad = zero(JuMP.GenericQuadExpr{Float64, GeneralVariableRef})
    # expand each variable independently and add all together
    for (var, coef) in expr.terms
        JuMP.add_to_expression!(quad, coef, _expand_measure(var, data,
                                                            trans_model,
                                                            point_mapper))
    end
    # expand over the constant
    if expr.constant != 0
        for i = 1:length(data.supports)
            JuMP.add_to_expression!(quad, data.coefficients[i] *
                                    data.weight_function(data.supports[i]) *
                                    expr.constant)
        end
    end
    # return affexpr if appropriate
    if length(quad.terms) == 0
        return quad.aff
    else
        return quad
    end
end

# GenericQuadExpr
function _expand_measure(expr::JuMP.GenericQuadExpr,
                         data::Union{DiscreteMeasureData,
                                     MultiDiscreteMeasureData},
                         trans_model::JuMP.AbstractModel,
                         point_mapper::Function)::JuMP.AbstractJuMPScalar
    quad = zero(JuMP.GenericQuadExpr{Float64, GeneralVariableRef})
    # convert the GenericAffExpr
    quad.aff = _expand_measure(expr.aff, data, trans_model, point_mapper)
    for (pair, coef) in expr.terms
        # expand on both variables
        expr_a = _expand_measure(pair.a, data, trans_model, point_mapper)
        expr_b = _expand_measure(pair.b, data, trans_model, point_mapper)
        vars_a = collect(keys(expr_a.terms))
        vars_b = collect(keys(expr_b.terms))
        alt_terms = false
        # check for case that a variable was a parameter converted to a number
        if length(vars_a) == 0
            vars_a = _get_param_value_list(pair.a, data)
            terms = [data.coefficients[i] * data.weight_function(data.supports[i]) for i = 1:length(data.supports)]
            alt_terms = true
        end
        if length(vars_b) == 0
            vars_b = _get_param_value_list(pair.b, data)
            terms = [data.coefficients[i] * data.weight_function(data.supports[i]) for i = 1:length(data.supports)]
            alt_terms = true
        end
        # combine both variable expressions using the coefficients from one
        # of them
        if length(vars_a) == length(vars_b)
            # are same length therefore have same coefficients
            for i = 1:length(vars_a)
                if alt_terms
                    JuMP.add_to_expression!(quad, coef * terms[i], vars_a[i],
                                            vars_b[i])
                else
                    JuMP.add_to_expression!(quad, coef * expr_a.terms[vars_a[i]],
                                            vars_a[i], vars_b[i])
                end
            end
        elseif length(vars_a) == 1
            # var_a was effectively a constant and var_b was't
            for i = 1:length(vars_b)
                if alt_terms
                    JuMP.add_to_expression!(quad, coef * terms[i], vars_a[1],
                                            vars_b[i])
                else
                    JuMP.add_to_expression!(quad, coef * expr_b.terms[vars_b[i]],
                                            vars_a[1], vars_b[i])
                end
            end
        else
            # var_b was effectively a constant and var_a was't
            for i = 1:length(vars_a)
                if alt_terms
                    JuMP.add_to_expression!(quad, coef * terms[i], vars_a[i],
                                            vars_b[1])
                else
                    JuMP.add_to_expression!(quad, coef * expr_a.terms[vars_a[i]],
                                            vars_a[i], vars_b[1])
                end
            end
        end
    end
    return quad
end

# MeasureRef
function _expand_measure(mref::MeasureRef,
                         data::Union{DiscreteMeasureData,
                                     MultiDiscreteMeasureData},
                         trans_model::JuMP.AbstractModel,
                         point_mapper::Function)::JuMP.AbstractJuMPScalar
    # determine function and data of the inner measure
    deeper_func = measure_function(mref)
    deeper_data = measure_data(mref)
    # expand the inner measure (note this is recursive for nested measures)
    new_func = _expand_measure(deeper_func, deeper_data, trans_model,
                               point_mapper)
    # expand current level with the inner measure now expanded
    return _expand_measure(new_func, data, trans_model, point_mapper)
end

# Catch all method for undefined behavior
function _expand_measure(expr, data, trans_model::JuMP.AbstractModel,
                         point_mapper::Function)
    expr_type = typeof(expr)
    data_type = typeof(data)
    error("Undefined behavior to expand expression of type $expr_type with " *
          "measure data $data_type. If this functionality is needed consider " *
          "extending `_expand_measure`.")
    return
end

# Map temp point variable references to actual variables by adding a variable to
# the infinite model. This is used by expand(mref) to expand measures within an
# infinite model
function _add_mapped_point_variable(model::InfiniteModel,
                                    pvref::PointVariableRef,
                                    ivref::InfiniteVariableRef, support::Tuple)
    # build variable
    var = JuMP.build_variable(error, _variable_info(ivref), Point,
                              infinite_variable_ref = ivref,
                              parameter_values = support)
    # add the variable completely (note the reference is already made)
    desired_index = JuMP.index(pvref)
    curr_index = model.next_var_index
    model.next_var_index = desired_index - 1
    JuMP.add_variable(model, var)
    model.next_var_index = curr_index
    return
end

"""
    expand(mref::MeasureRef)::JuMP.AbstractJuMPScalar

Return a JuMP scalar function containing the explicit expansion of the measure
`mref`. This expansion is done according to the measure data. Note that
variables are added to the model as necessary to accomodate the expansion (i.e.,
point variables and reduced infinite variables are made as needed). Errors if
expansion is undefined for the measure data and/or the measure expression. If
desired this can be used in combination with [`measure`](@ref) to expand measures
on the fly.

This is useful for extensions that employ a custom optimizer_model since it
can be used evaluate measures before expressions are translated to the new model.
This method can also be extended to handle custom measure data types by extending
`InfiniteOpt._expand_measure` which should be of the form
`InfiniteOpt._expand_measure(::AbstractJuMPScalar, ::AbstractMeasureData, ::InfiniteModel, point_mapper::Function)`.
See the source code in InfiniteOpt/src/measures.jl for examples of how to do this.

**Example**
```julia
julia> tdata = DiscreteMeasureData(t, [0.5, 0.5], [0, 1])

julia> expr = expand(measure(g + z + T - h - 2, tdata))
0.5 g(0) + 0.5 g(1) + z + 0.5 T(0, x) + 0.5 T(1, x) - h(x) - 2
```
"""
function expand(mref::MeasureRef)::JuMP.AbstractJuMPScalar
    return _expand_measure(measure_function(mref), measure_data(mref),
                           JuMP.owner_model(mref), _add_mapped_point_variable)
end

## Helper functions for expanding the measure references in expressions
# MeasureRef
function _expand_measures(mref::MeasureRef,
                          expand_model::JuMP.AbstractModel,
                          point_mapper::Function)::JuMP.AbstractJuMPScalar
    return _expand_measure(measure_function(mref), measure_data(mref),
                           expand_model, point_mapper)
end

# GenericAffExpr
function _expand_measures(expr::JuMP.GenericAffExpr{C, <:GeneralVariableRef},
                          expand_model::JuMP.AbstractModel,
                          point_mapper::Function)::JuMP.AbstractJuMPScalar where {C}
    # use a QuadExpr in case measures contain quadratic espressions
    quad = zero(JuMP.GenericQuadExpr{C, GeneralVariableRef})
    quad.aff.constant = expr.constant
    # add the variables to the expr, converting measures into expanded exprs
    for (var, coef) in expr.terms
        if isa(var, MeasureRef)
            new_func = _expand_measure(measure_function(var), measure_data(var),
                                       expand_model, point_mapper)
            JuMP.add_to_expression!(quad, coef, new_func)
        else
            JuMP.add_to_expression!(quad, coef, var)
        end
    end
    # return a AffExpr if there are no quadratic terms
    if length(quad.terms) == 0
        return quad.aff
    else
        return quad
    end
end

# GenericQuadExpr
function _expand_measures(expr::JuMP.GenericQuadExpr{C, <:GeneralVariableRef},
                          expand_model::JuMP.AbstractModel,
                          point_mapper::Function)::JuMP.AbstractJuMPScalar where {C}
    quad = zero(JuMP.GenericQuadExpr{C, GeneralVariableRef})
    quad.aff = _expand_measures(expr.aff, expand_model, point_mapper)
    # add the quadratic terms to the expr, converting measures into expanded exprs
    # note that this will error if the expanded forms are not linear or quadratic
    for (pair, coef) in expr.terms
        var_a = pair.a
        var_b = pair.b
        if isa(var_a, MeasureRef)
            var_a = _expand_measure(measure_function(var_a), measure_data(var_a),
                                    expand_model, point_mapper)
        end
        if isa(var_b, MeasureRef)
            var_b = _expand_measure(measure_function(var_b), measure_data(var_b),
                                    expand_model, point_mapper)
        end
        JuMP.add_to_expression!(quad, coef * var_a * var_b)
    end
    return quad
end

"""
    expand_all_measures!(model::InfiniteModel)

Expand all of the measures used in the objective and/or constraints of `model`.
The objective and constraints are updated accordingly. Note that
variables are added to the model as necessary to accomodate the expansion (i.e.,
point variables and reduced infinite variables are made as needed). Errors if
expansion is undefined for the measure data and/or the measure expression. Also
errors if the expanded objective function is not finite.

This is useful for extensions that employ a custom optimizer_model since it
can be used evaluate measures before `model` is translated into the new model.
This method can also be extended to handle custom measure data types by extending
`InfiniteOpt._expand_measure` which should be of the form
`InfiniteOpt._expand_measure(::AbstractJuMPScalar, ::AbstractMeasureData, ::InfiniteModel, point_mapper::Function)`.
See the source code in InfiniteOpt/src/measures.jl for examples of how to do this.

**Example**
```julia
julia> print(model)
Min measure(g(t)*t) + z
Subject to
 T(t, xi) >= 0.0
 z >= 0.0
 g(t) + z >= 42.0
 measure(T(t, xi)) >= 0.0, for all xi in [-1, 1]
 t in [0, 6]
 xi in Normal(μ=0.0, σ=1.0)

julia> expand_all_measures!(model)

julia> print(model)
Min 3 g(6) + z
Subject to
 T(t, xi) >= 0.0
 z >= 0.0
 g(t) + z >= 42.0
 0.5 T(0, xi) + 0.5 T(6, xi) >= 0.0, for all xi in [-1, 1]
 t in [0, 6]
 xi in Normal(μ=0.0, σ=1.0)
```
"""
function expand_all_measures!(model::InfiniteModel)
    # expand the objective if it contains measures
    if JuMP.objective_function_type(model) <: MeasureExpr
        new_obj = _possible_convert(FiniteVariableRef,
                         _expand_measures(JuMP.objective_function(model), model,
                                          _add_mapped_point_variable))
        isa(new_obj, InfiniteExpr) && error("Objective is not finite, ensure " *
                                            "all infinite variables/parameters " *
                                            "in measures are evaluated " *
                                            "completely.")
        JuMP.set_objective(model, JuMP.objective_sense(model), new_obj)
    end
    # extract indexes of constraints that contain measures
    cindices = Int[]
    for list in values(model.meas_to_constrs)
        cindices = [cindices; list]
    end
    # expand all of the constraints that contain measures
    for cindex in unique(cindices)
        # expand the expression
        new_func = _possible_convert(FiniteVariableRef,
                                 _expand_measures(model.constrs[cindex].func,
                                                  model,
                                                  _add_mapped_point_variable))
        # get the necessary info
        cref = _make_constraint_ref(model, cindex)
        name = JuMP.name(cref)
        set = model.constrs[cindex].set
        curr_index = model.next_constr_index
        # delete the old cosntraint and replace it with the expanded version
        model.next_constr_index = cindex - 1
        if isa(model.constrs[cindex], BoundedScalarConstraint) && isa(new_func,
                                                                   InfiniteExpr)
            bounds = model.constrs[cindex].bounds
            JuMP.delete(model, cref)
            JuMP.add_constraint(model, JuMP.build_constraint(error, new_func,
                                set, parameter_bounds = bounds), name)
        else
            juMP.delete(model, cref)
            JuMP.add_constraint(model, JuMP.build_constraint(error, new_func,
                                set), name)
        end
        model.next_constr_index = curr_index
    end
    return
end
