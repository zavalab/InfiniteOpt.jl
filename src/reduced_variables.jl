# Helper function to get reduced variable info
function _reduced_info(vref::InfOptVariableRef)::ReducedInfiniteInfo
    return _reduced_info(vref, Val(variable_type(vref)))
end

function _reduced_info(vref::InfOptVariableRef, ::Val{Reduced})::ReducedInfiniteInfo
    return JuMP.owner_model(vref).reduced_info[JuMP.index(vref)]
end

"""
    infinite_variable_ref(vref::InfOptVariableRef)::InfOptVariableRef

Return the infinite variable referece associated with the reduced infinite
variable `vref`.

**Example**
```julia
julia> infinite_variable_ref(vref)
g(t, x)
```
"""
function infinite_variable_ref(vref::InfOptVariableRef)::InfOptVariableRef
    return infinite_variable_ref(vref, Val(variable_type(vref)))
end

function infinite_variable_ref(vref::InfOptVariableRef,
                               ::Val{Reduced})::InfOptVariableRef
    return _reduced_info(vref).infinite_variable_ref
end

"""
    eval_supports(vref::InfOptVariableRef)::Dict

Return the evaluation supports associated with the reduced infinite variable
`vref`.

**Example**
```julia
julia> eval_supports(vref)
Dict{Int64,Float64} with 1 entry:
  1 => 0.5
```
"""
function eval_supports(vref::InfOptVariableRef)::Dict
    return eval_supports(vref, Val(variable_type(vref)))
end

function eval_supports(vref::InfOptVariableRef, ::Val{Reduced})::Dict
    return _reduced_info(vref).eval_supports
end

"""
    parameter_refs(vref::InfOptVariableRef)::Tuple

Return the parameter references associated with the reduced infinite variable
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
function parameter_refs(vref::InfOptVariableRef)::Tuple
    return parameter_refs(vref, Val(variable_type(vref)))
end

function parameter_refs(vref::InfOptVariableRef, ::Val{Reduced})::Tuple
    orig_prefs = parameter_refs(infinite_variable_ref(vref))
    prefs = Tuple(orig_prefs[i] for i = 1:length(orig_prefs) if !haskey(eval_supports(vref), i))
    return prefs
end

# JuMP.name for reduced variables
function JuMP.name(vref::InfOptVariableRef, ::Val{Reduced})::String
    root_name = _root_name(infinite_variable_ref(vref))
    prefs = parameter_refs(infinite_variable_ref(vref))
    param_names = [_root_name(first(pref)) for pref in prefs]
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

# JuMP.has_lower_bound for reduced variables
function JuMP.has_lower_bound(vref::InfOptVariableRef, ::Val{Reduced})::Bool
    return JuMP.has_lower_bound(infinite_variable_ref(vref))
end

# JuMP.lower_bound for reduced variables
function JuMP.lower_bound(vref::InfOptVariableRef, ::Val{Reduced})::Float64
    if !JuMP.has_lower_bound(vref)
        error("Variable $(vref) does not have a lower bound.")
    end
    return JuMP.lower_bound(infinite_variable_ref(vref))
end

# Extend to return the index of the lower bound constraint associated with the
# original infinite variable of `vref`.
#=
function JuMP._lower_bound_index(vref::InfOptVariableRef)::Int64
    return JuMP._lower_bound_index(vref, Val(variable_type(vref)))
end

function JuMP._lower_bound_index(vref::InfOptVariableRef, ::Val{Reduced})::Int64
    if !JuMP.has_lower_bound(vref)
        error("Variable $(vref) does not have a lower bound.")
    end
    return JuMP._lower_bound_index(infinite_variable_ref(vref))
end

# JuMP.LowerBoundRef for reduced variables
function JuMP.LowerBoundRef(vref::InfOptVariableRef,
                            ::Val{Reduced})::InfOptConstraintRef
    return JuMP.LowerBoundRef(infinite_variable_ref(vref), Val(Infinite))
end
=#
# JuMP.has_upper_bound for reduced variables
function JuMP.has_upper_bound(vref::InfOptVariableRef, ::Val{Reduced})::Bool
    return JuMP.has_upper_bound(infinite_variable_ref(vref))
end

# JuMP.upper_bound for reduced variables
function JuMP.upper_bound(vref::InfOptVariableRef, ::Val{Reduced})::Float64
    if !JuMP.has_upper_bound(vref)
        error("Variable $(vref) does not have a upper bound.")
    end
    return JuMP.upper_bound(infinite_variable_ref(vref))
end

"""
    JuMP.UpperBoundRef(vref::InfOptVariableRef)::InfOptConstraintRef

Extend [`JuMP.UpperBoundRef`](@ref) to extract a constraint reference for the
upper bound of the original infinite variable of `vref`.

**Example**
```julia
julia> cref = UpperBoundRef(vref)
var <= 1.0
```
"""

function JuMP.UpperBoundRef(vref::InfOptVariableRef,
                            ::Val{Reduced})::InfOptConstraintRef
    return JuMP.UpperBoundRef(infinite_variable_ref(vref))
end

"""
    JuMP.is_fixed(vref::InfOptVariableRef)::Bool

Extend [`JuMP.is_fixed`](@ref) to return `Bool` whether the original infinite
variable of `vref` is fixed.

**Example**
```julia
julia> is_fixed(vref)
true
```
"""

# JuMP.fix_value for reduced variable refs
#=
function JuMP.fix_value(vref::InfOptVariableRef, ::Val{Reduced})::Float64
    if !JuMP.is_fixed(vref)
        error("Variable $(vref) is not fixed.")
    end
    return JuMP.fix_value(infinite_variable_ref(vref))
end

# JuMP._fix_index for reduced variable refs
function JuMP._fix_index(vref::InfOptVariableRef, ::Val{Reduced})::Int64
    if !JuMP.is_fixed(vref)
        error("Variable $(vref) is not fixed.")
    end
    return JuMP._fix_index(infinite_variable_ref(vref))
end

# JuMP.FixRef for reduced variable refs
function JuMP.FixRef(vref::InfOptVariableRef,
                     ::Val{Reduced})::InfOptConstraintRef
    return JuMP.FixRef(infinite_variable_ref(vref))
end
=#
"""
    JuMP.start_value(vref::InfOptVariableRef)::Union{Nothing, Float64}

Extend [`JuMP.start_value`](@ref) to return starting value of the original
infinite variable of `vref` if it has one. Returns `nothing` otherwise.

**Example**
```julia
julia> start_value(vref)
0.0
```
"""

function JuMP.start_value(vref::InfOptVariableRef, ::Val{Reduced})::Union{Nothing, Float64}
    return JuMP.start_value(infinite_variable_ref(vref))
end

"""
    JuMP.is_binary(vref::InfOptVariableRef)::Bool

Extend [`JuMP.is_binary`](@ref) to return `Bool` whether the original infinite
variable of `vref` is binary.

**Example**
```julia
julia> is_binary(vref)
true
```
"""
# JuMP.is_binary for reduced variable refs
function JuMP.is_binary(vref::InfOptVariableRef, ::Val{Reduced})::Bool
    return JuMP.is_binary(infinite_variable_ref(vref))
end

# JuMP._binary_index for reduced variable refs
function JuMP._binary_index(vref::InfOptVariableRef, ::Val{Reduced})::Int64
    if !JuMP.is_binary(vref)
        error("Variable $(vref) is not binary.")
    end
    return JuMP._binary_index(infinite_variable_ref(vref))
end

"""
    JuMP.BinaryRef(vref::InfOptVariableRef)::InfOptConstraintRef

Extend [`JuMP.BinaryRef`](@ref) to return a constraint reference to the
constraint constrainting the original infinite variable of `vref` to be binary.
Errors if one does not exist.

**Example**
```julia
julia> cref = BinaryRef(vref)
var binary
```
"""
function JuMP.BinaryRef(vref::InfOptVariableRef)::InfOptConstraintRef
    return JuMP.BinaryRef(vref, Val(variable_type(vref)))
end

function JuMP.BinaryRef(vref::InfOptVariableRef, ::Val{Reduced})::InfOptConstraintRef
    return JuMP.BinaryRef(infinite_variable_ref(vref))
end

# JuMP.is_integer for reduced variable refs
function JuMP.is_integer(vref::InfOptVariableRef, ::Val{Reduced})::Bool
    return JuMP.is_integer(infinite_variable_ref(vref))
end

# JuMP._integer_index for reduced variable refs
function JuMP._integer_index(vref::InfOptVariableRef, ::Val{Reduced})::Int64
    if !JuMP.is_integer(vref)
        error("Variable $(vref) is not an integer.")
    end
    return JuMP._integer_index(infinite_variable_ref(vref))
end

# JuMP.IntegerRef for reduced variable refs
function JuMP.IntegerRef(vref::InfOptVariableRef, ::Val{Reduced})::InfOptConstraintRef
    return JuMP.IntegerRef(infinite_variable_ref(vref))
end

# used_by_constraint for reduced variables
function used_by_constraint(vref::InfOptVariableRef, ::Val{Reduced})::Bool
    return haskey(JuMP.owner_model(vref).reduced_to_constrs, JuMP.index(vref))
end

# used_by_measure for reduced variables
function used_by_measure(vref::InfOptVariableRef, ::Val{Reduced})::Bool
    return haskey(JuMP.owner_model(vref).reduced_to_meas, JuMP.index(vref))
end

# JuMP.is_valid for reduced variables
function JuMP.is_valid(model::InfiniteModel, vref::InfOptVariableRef,
                       ::Val{Reduced})::Bool
    return (model === JuMP.owner_model(vref) && JuMP.index(vref) in keys(model.reduced_info))
end

# JuMP.delete for reduced variables
function JuMP.delete(model::InfiniteModel, vref::InfOptVariableRef, ::Val{Reduced})
    # check valid reference
    @assert JuMP.is_valid(model, vref) "Invalid variable reference."
    # remove from measures if used
    if used_by_measure(vref)
        for mindex in model.reduced_to_meas[JuMP.index(vref)]
            if isa(model.measures[mindex].func, InfOptVariableRef) &&
               variable_type(model.measures[mindex].func) == Reduced
                model.measures[mindex] = Measure(zero(JuMP.AffExpr),
                                                 model.measures[mindex].data)
            else
                _remove_variable(model.measures[mindex].func, vref)
            end
            JuMP.set_name(InfOptVariableRef(model, mindex, MeasureRef),
                          _make_meas_name(model.measures[mindex]))
        end
        # delete mapping
        delete!(model.reduced_to_meas, JuMP.index(vref))
    end
    # remove from constraints if used
    if used_by_constraint(vref)
        for cindex in model.reduced_to_constrs[JuMP.index(vref)]
            if isa(model.constrs[cindex].func, InfOptVariableRef) &&
               variable_type(model.constrs[cindex].func) == Reduced
                model.constrs[cindex] = JuMP.ScalarConstraint(zero(JuMP.AffExpr),
                                                      model.constrs[cindex].set)
            else
                _remove_variable(model.constrs[cindex].func, vref)
            end
        end
        # delete mapping
        delete!(model.reduced_to_constrs, JuMP.index(vref))
    end
    # remove mapping to infinite variable
    ivref = infinite_variable_ref(vref)
    filter!(e -> e != JuMP.index(vref),
            model.infinite_to_reduced[JuMP.index(ivref)])
    if length(model.infinite_to_reduced[JuMP.index(ivref)]) == 0
        delete!(model.infinite_to_reduced, JuMP.index(ivref))
    end
    # delete the info
    delete!(model.reduced_info, JuMP.index(vref))
    return
end
