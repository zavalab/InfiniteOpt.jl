"""
    JuMP.owner_model(cref::InfOptConstraintRef)::InfiniteModel

Extend [`JuMP.owner_model`](@ref) to return the infinite model associated with
`cref`.

**Example**
```julia
julia> model = owner_model(cref)
An InfiniteOpt Model
Minimization problem with:
Variables: 3
Objective function type: GlobalVariableRef
`GenericAffExpr{Float64,FiniteVariableRef}`-in-`MathOptInterface.EqualTo{Float64}`: 1 constraint
Names registered in the model: g, t, h, x
Optimizer model backend information:
Model mode: AUTOMATIC
CachingOptimizer state: NO_OPTIMIZER
Solver name: No optimizer attached.
```
"""
JuMP.owner_model(cref::InfOptConstraintRef)::InfiniteModel = cref.model

"""
    JuMP.index(cref::InfOptConstraintRef)::Int64

Extend [`JuMP.index`](@ref) to return the index of an `InfiniteOpt` constraint
`cref`.

**Example**
```julia
julia> index(cref)
2
```
"""
JuMP.index(cref::InfOptConstraintRef)::Int64 = cref.index

# Extend Base and JuMP functions
function Base.:(==)(v::InfOptConstraintRef, w::InfOptConstraintRef)::Bool
    return v.model === w.model && v.index == w.index && v.shape == w.shape && v.type == w.type
end
Base.broadcastable(cref::InfOptConstraintRef) = Ref(cref)
JuMP.constraint_type(m::InfiniteModel) = InfOptConstraintRef

# Set default bounded set
const default_bounds = Dict{InfOptVariableRef, IntervalSet}()

# This might not be necessary...
function JuMP.build_constraint(_error::Function,
                               v::InfOptVariableRef,
                               set::MOI.AbstractScalarSet;
                               parameter_bounds::Dict{InfOptVariableRef,
                                                  IntervalSet} = default_bounds)
    if length(parameter_bounds) != 0
        return BoundedScalarConstraint(v, set, parameter_bounds)
    else
        return JuMP.ScalarConstraint(v, set)
    end
end

"""
    JuMP.build_constraint(_error::Function, expr::Expr,
                          set::MOI.AbstractScalarSet;
                          [parameter_bounds::Dict{InfOptVariableRef,
                                                  IntervalSet} = Dict()])

Extend [`JuMP.build_constraint`](@ref) to accept the parameter_bounds argument
and return a [`BoundedScalarConstraint`](@ref) if the `parameter_bounds` keyword
argument is specifed or return a [`ScalarConstraint`](@ref) otherwise. This is
primarily intended to work as an internal function for constraint macros.

**Example**
```julia
julia> constr = build_constraint(error, g + x, MOI.EqualTo(42.0),
                               parameter_bounds = Dict(t => IntervalSet(0, 1)));

julia> isa(constr, BoundedScalarConstraint)
true
```
"""
function JuMP.build_constraint(_error::Function, expr::Expr,
                               set::MOI.AbstractScalarSet;
                               parameter_bounds::Dict{InfOptVariableRef,
                                                  IntervalSet} = default_bounds)
    offset = JuMP.constant(expr)
    JuMP.add_to_expression!(expr, -offset)
    if length(parameter_bounds) != 0
        return BoundedScalarConstraint(expr, MOIU.shift_constant(set, -offset),
                                       parameter_bounds)
    else
        return JuMP.ScalarConstraint(expr, MOIU.shift_constant(set, -offset))
    end
end

# Used to update the model.var_to_constrs field
function _update_var_constr_mapping(vrefs::Vector{InfOptVariableRef},
                                    cindex::Int64)
    for vref in vrefs
        model = JuMP.owner_model(vref)
        if isa(Val(variable_type(vref)), Variables)
            if haskey(model.var_to_constrs, JuMP.index(vref))
                push!(model.var_to_constrs[JuMP.index(vref)], cindex)
            else
                model.var_to_constrs[JuMP.index(vref)] = [cindex]
            end
        elseif variable_type(vref) == Parameter
            if haskey(model.param_to_constrs, JuMP.index(vref))
                push!(model.param_to_constrs[JuMP.index(vref)], cindex)
            else
                model.param_to_constrs[JuMP.index(vref)] = [cindex]
            end
        elseif variable_type(vref) == MeasureRef
            if haskey(model.meas_to_constrs, JuMP.index(vref))
                push!(model.meas_to_constrs[JuMP.index(vref)], cindex)
            else
                model.meas_to_constrs[JuMP.index(vref)] = [cindex]
            end
        elseif variable_type(vref) == Reduced
            if haskey(model.reduced_to_constrs, JuMP.index(vref))
                push!(model.reduced_to_constrs[JuMP.index(vref)], cindex)
            else
                model.reduced_to_constrs[JuMP.index(vref)] = [cindex]
            end
        end
    end
    return
end

# Check that parameter_bounds argument is valid
function _check_bounds(model::InfiniteModel, bounds::Dict)
    prefs = collect(keys(bounds))
    for pref in prefs
        !JuMP.is_valid(model, pref) && error("Parameter bound reference " *
                                             "is invalid.")
        if JuMP.has_lower_bound(pref)
            if bounds[pref].lower_bound < JuMP.lower_bound(pref)
                error("Specified parameter lower bound exceeds that defined " *
                      "for $pref.")
            end
        end
        if JuMP.has_upper_bound(pref)
            if bounds[pref].upper_bound > JuMP.upper_bound(pref)
                error("Specified parameter upper bound exceeds that defined " *
                      "for $pref.")
            end
        end
    end
    return
end

# Extend functions for bounded constraints
JuMP.shape(c::BoundedScalarConstraint) = JuMP.shape(JuMP.ScalarConstraint(c.func, c.set))
JuMP.jump_function(c::BoundedScalarConstraint) = c.func
JuMP.moi_set(c::BoundedScalarConstraint) = c.set

"""
    JuMP.add_constraint(model::InfiniteModel, c::JuMP.AbstractConstraint,
                        [name::String = ""])

Extend [`JuMP.add_constraint`](@ref) to add a constraint `c` to an infinite model
`model` with name `name`. Returns an appropriate constraint reference whose type
depends on what variables are used to define the constraint. Errors if a vector
constraint is used, the constraint only constains parameters, or if any
variables do not belong to `model`. This is primarily used as an internal
method for the cosntraint macros.

**Example**
```julia
julia> constr = build_constraint(error, g + x, MOI.EqualTo(42));

julia> cref = add_constraint(model, constr, "name")
name : g(t) + x == 42.0
```
"""
function JuMP.add_constraint(model::InfiniteModel, c::JuMP.AbstractConstraint,
                             name::String = "")
    isa(c, JuMP.VectorConstraint) && error("Vector constraints not supported.")
    JuMP.check_belongs_to_model(c.func, model)
    vrefs = _all_function_variables(c.func)
    isa(vrefs, Vector{InfOptVariableRef}) && error("Constraints cannot contain " *
                                                   "only parameters.")
    if isa(c, BoundedScalarConstraint)
        _check_bounds(model, c.bounds)
    end
    model.next_constr_index += 1
    index = model.next_constr_index
    if length(vrefs) != 0
        _update_var_constr_mapping(vrefs, index)
    end
    if _is_infinite_expr(c.func)
        cref = InfOptConstraintRef(model, index, JuMP.shape(c), Infinite)
    elseif _is_measure_expr(c.func)
        cref = InfOptConstraintRef(model, index, JuMP.shape(c), MeasureRef)
    elseif _is_single_type_expr(c.func, Global)
        cref = InfOptConstraintRef(model, index, JuMP.shape(c), Global)
    elseif _is_single_type_expr(c.func, Point)
        cref = InfOptConstraintRef(model, index, JuMP.shape(c), Point)
    else
        cref = InfOptConstraintRef(model, index, JuMP.shape(c), Finite)
    end
    model.constrs[index] = c
    JuMP.set_name(cref, name)
    model.constr_in_var_info[index] = false
    set_optimizer_model_ready(model, false)
    return cref
end

"""
    JuMP.delete(model::InfiniteModel, cref::InfOptConstraintRef)

Extend [`JuMP.delete`](@ref) to delete an `InfiniteOpt` constraint and all
associated information. Errors if `cref` is invalid.

**Example**
```julia
julia> print(model)
Min measure(g(t)*t) + z
Subject to
 z >= 0.0
 g(t) + z >= 42.0
 t in [0, 6]

julia> delete(model, cref)

julia> print(model)
Min measure(g(t)*t) + z
Subject to
 z >= 0.0
 t in [0, 6]
```
"""
function JuMP.delete(model::InfiniteModel, cref::InfOptConstraintRef)
    # check valid reference
    @assert JuMP.is_valid(model, cref) "Invalid constraint reference."
    # update variable dependencies
    all_vrefs = _all_function_variables(model.constrs[JuMP.index(cref)].func)
    for vref in all_vrefs
        if isa(Val(variable_type(vref)), Variables)
            filter!(e -> e != JuMP.index(cref),
                    model.var_to_constrs[JuMP.index(vref)])
            if length(model.var_to_constrs[JuMP.index(vref)]) == 0
                delete!(model.var_to_constrs, JuMP.index(vref))
            end
        elseif variable_type(vref) == Parameter
            filter!(e -> e != JuMP.index(cref),
                    model.param_to_constrs[JuMP.index(vref)])
            if length(model.param_to_constrs[JuMP.index(vref)]) == 0
                delete!(model.param_to_constrs, JuMP.index(vref))
            end
        elseif variable_type(vref) == MeasureRef
            filter!(e -> e != JuMP.index(cref),
                    model.meas_to_constrs[JuMP.index(vref)])
            if length(model.meas_to_constrs[JuMP.index(vref)]) == 0
                delete!(model.meas_to_constrs, JuMP.index(vref))
            end
        elseif variable_type(vref) == Reduced
            filter!(e -> e != JuMP.index(cref),
                    model.reduced_to_constrs[JuMP.index(vref)])
            if length(model.reduced_to_constrs[JuMP.index(vref)]) == 0
                delete!(model.reduced_to_constrs, JuMP.index(vref))
            end
        end
    end
    # delete constraint information
    delete!(model.constrs, JuMP.index(cref))
    delete!(model.constr_to_name, JuMP.index(cref))
    delete!(model.constr_in_var_info, JuMP.index(cref))
    # reset optimizer model status
    set_optimizer_model_ready(model, false)
    return
end

"""
    JuMP.is_valid(model::InfiniteModel, cref::InfOptConstraintRef)::Bool

Extend [`JuMP.is_valid`](@ref) to return `Bool` whether an `InfiniteOpt`
constraint reference is valid.

**Example**
```julia
julia> is_valid(model, cref)
true
```
"""
function JuMP.is_valid(model::InfiniteModel, cref::InfOptConstraintRef)::Bool
    return (model === JuMP.owner_model(cref) && JuMP.index(cref) in keys(model.constrs))
end

"""
    JuMP.constraint_object(cref::InfOptConstraintRef)::JuMP.AbstractConstraint

Extend [`JuMP.constraint_object`](@ref) to return the constraint object
associated with `cref`.

**Example**
```julia
julia> obj = constraint_object(cref)
ScalarConstraint{GlobalVariableRef,MathOptInterface.LessThan{Float64}}(x,
MathOptInterface.LessThan{Float64}(1.0))
```
"""
function JuMP.constraint_object(cref::InfOptConstraintRef)::JuMP.AbstractConstraint
    return JuMP.owner_model(cref).constrs[JuMP.index(cref)]
end

"""
    JuMP.name(cref::InfOptConstraintRef)::String

Extend [`JuMP.name`](@ref) to return the name of an `InfiniteOpt` constraint.

**Example**
```julia
julia> name(cref)
constr_name
```
"""
function JuMP.name(cref::InfOptConstraintRef)::String
    return JuMP.owner_model(cref).constr_to_name[JuMP.index(cref)]
end

"""
    JuMP.set_name(cref::InfOptConstraintRef, name::String)

Extend [`JuMP.set_name`](@ref) to specify the name of a constraint `cref`.

**Example**
```julia
julia> set_name(cref, "new_name")

julia> name(cref)
new_name
```
"""
function JuMP.set_name(cref::InfOptConstraintRef, name::String)
    JuMP.owner_model(cref).constr_to_name[JuMP.index(cref)] = name
    JuMP.owner_model(cref).name_to_constr = nothing
    return
end

# Return a constraint set with an updated value
function _set_set_value(set::S, value::Real) where {T, S <: Union{MOI.LessThan{T},
                                            MOI.GreaterThan{T}, MOI.EqualTo{T}}}
    return S(convert(T, value))
end

"""
    JuMP.set_normalized_rhs(cref::InfOptConstraintRef, value::Real)

Set the right-hand side term of `constraint` to `value`.
Note that prior to this step, JuMP will aggregate all constant terms onto the
right-hand side of the constraint. For example, given a constraint `2x + 1 <=
2`, `set_normalized_rhs(con, 4)` will create the constraint `2x <= 4`, not `2x +
1 <= 4`.

```julia
julia> @constraint(model, con, 2x + 1 <= 2)
con : 2 x <= 1.0

julia> set_normalized_rhs(con, 4)

julia> con
con : 2 x <= 4.0
```
"""
function JuMP.set_normalized_rhs(cref::InfOptConstraintRef, value::Real)
    old_constr = JuMP.constraint_object(cref)
    new_set = _set_set_value(old_constr.set, value)
    if old_constr isa BoundedScalarConstraint
        new_constr = BoundedScalarConstraint(old_constr.func, new_set,
                                             old_constr.bounds)
    else
        new_constr = JuMP.ScalarConstraint(old_constr.func, new_set)
    end
    JuMP.owner_model(cref).constrs[JuMP.index(cref)] = new_constr
    return
end

"""
    JuMP.normalized_rhs(cref::InfOptConstraintRef)::Number

Return the right-hand side term of `cref` after JuMP has converted the
constraint into its normalized form. See also [`set_normalized_rhs`](@ref).
"""
function JuMP.normalized_rhs(cref::InfOptConstraintRef)::Number
    con = JuMP.constraint_object(cref)
    return MOI.constant(con.set)
end

"""
    JuMP.add_to_function_constant(cref::InfOptConstraintRef, value::Real)

Add `value` to the function constant term.
Note that for scalar constraints, JuMP will aggregate all constant terms onto the
right-hand side of the constraint so instead of modifying the function, the set
will be translated by `-value`. For example, given a constraint `2x <=
3`, `add_to_function_constant(c, 4)` will modify it to `2x <= -1`.
```
"""
function JuMP.add_to_function_constant(cref::InfOptConstraintRef, value::Real)
    current_value = JuMP.normalized_rhs(cref)
    JuMP.set_normalized_rhs(cref, current_value - value)
    return
end

"""
    JuMP.set_normalized_coefficient(cref::InfOptConstraintRef,
                                    variable::InfOptVariableRef, value::Real)

Set the coefficient of `variable` in the constraint `constraint` to `value`.
Note that prior to this step, JuMP will aggregate multiple terms containing the
same variable. For example, given a constraint `2x + 3x <= 2`,
`set_normalized_coefficient(con, x, 4)` will create the constraint `4x <= 2`.

```julia
julia> con
con : 5 x <= 2.0

julia> set_normalized_coefficient(con, x, 4)

julia> con
con : 4 x <= 2.0
```
"""
function JuMP.set_normalized_coefficient(cref::InfOptConstraintRef,
                                         variable::InfOptVariableRef,
                                         value::Real)
    # update the constraint expression and update the constraint
    old_constr = JuMP.constraint_object(cref)
    new_expr = _set_variable_coefficient!(old_constr.func, variable, value)
    if old_constr isa BoundedScalarConstraint
        new_constr = BoundedScalarConstraint(new_expr, old_constr.set,
                                             old_constr.bounds)
    else
        new_constr = JuMP.ScalarConstraint(new_expr, old_constr.set)
    end
    JuMP.owner_model(cref).constrs[JuMP.index(cref)] = new_constr
    return
end

"""
    JuMP.normalized_coefficient(cref::InfOptConstraintRef,
                                variable::InfOptVariableRef)::Number

Return the coefficient associated with `variable` in `constraint` after JuMP has
normalized the constraint into its standard form. See also
[`set_normalized_coefficient`](@ref).
"""
function JuMP.normalized_coefficient(cref::InfOptConstraintRef,
                                     variable::InfOptVariableRef)::Number
    con = JuMP.constraint_object(cref)
    if con.func isa InfOptVariableRef && con.func == variable
        return 1.0
    elseif con.func isa InfOptVariableRef
        return 0.0
    else
        return JuMP._affine_coefficient(con.func, variable)
    end
end

# Return the appropriate constraint reference given the index and model
function _make_constraint_ref(model::InfiniteModel,
                              index::Int64)::InfOptConstraintRef
    if _is_infinite_expr(model.constrs[index].func)
        return InfOptConstraintRef(model, index,
                                    JuMP.shape(model.constrs[index]), Infinite)
    elseif _is_measure_expr(model.constrs[index].func)
        return InfOptConstraintRef(model, index,
                                    JuMP.shape(model.constrs[index]), MeasureRef)
    elseif _is_single_type_expr(model.constrs[index].func, Global)
        return InfOptConstraintRef(model, index,
                                    JuMP.shape(model.constrs[index]), Global)

    elseif _is_single_type_expr(model.constrs[index].func, Point)
        return InfOptConstraintRef(model, index,
                                    JuMP.shape(model.constrs[index]), Point)
    else
        return InfOptConstraintRef(model, index,
                                    JuMP.shape(model.constrs[index]), Finite)
    end
end

"""
    JuMP.constraint_by_name(model::InfiniteModel,
                            name::String)::Union{InfOptConstraintRef, Nothing}

Extend [`JuMP.constraint_by_name`](@ref) to return the constraint reference
associated with `name` if one exists or returns nothing. Errors if more than
one constraint uses the same name.

**Example**
```julia
julia> constraint_by_name(model, "constr_name")
constr_name : x + pt == 3.0
```
"""
function JuMP.constraint_by_name(model::InfiniteModel, name::String)
    if model.name_to_constr === nothing
        # Inspired from MOI/src/Utilities/model.jl
        model.name_to_constr = Dict{String, Int}()
        for (constr, constr_name) in model.constr_to_name
            if haskey(model.name_to_constr, constr_name)
                # -1 is a special value that means this string does not map to
                # a unique constraint name.
                model.name_to_constr[constr_name] = -1
            else
                model.name_to_constr[constr_name] = constr
            end
        end
    end
    index = get(model.name_to_constr, name, nothing)
    if index isa Nothing
        return nothing
    elseif index == -1
        error("Multiple constraints have the name $name.")
    else
        return _make_constraint_ref(model, index)
    end
end

"""
    JuMP.num_constraints(model::InfiniteModel,
                         function_type::Type{<:Union{Symbol, JuMP.AbstractJuMPScalar}},
                         set_type::Type{<:MOI.AbstractSet})::Int64

Extend [`JuMP.num_constraints`](@ref) to return the number of constraints
with a partiuclar function or variable type and set type.

**Example**
```julia
julia> num_constraints(model, GlobalVariableRef, MOI.LessThan)
1
```
"""
function JuMP.num_constraints(model::InfiniteModel, function_type::Symbol,
                              set_type::Type{<:MOI.AbstractSet})::Int64
    counter = 0
    for (index, constr) in model.constrs
        if isa(constr.func, InfOptVariableRef) &&
           variable_type(constr.func) == function_type &&
           isa(constr.set, set_type)
            counter += 1
        end
    end
    return counter
end

function JuMP.num_constraints(model::InfiniteModel,
                              function_type::Type{<:JuMP.AbstractJuMPScalar},
                              set_type::Type{<:MOI.AbstractSet})::Int64
    counter = 0
    for (index, constr) in model.constrs
        if isa(constr.func, function_type) && isa(constr.set, set_type)
            counter += 1
        end
    end
    return counter
end

"""
    JuMP.num_constraints(model::InfiniteModel,
                         function_type::Type{<:JuMP.AbstractJuMPScalar})::Int64

Extend [`JuMP.num_constraints`](@ref) to search by function types for all MOI
sets and return the total number of constraints with a particular function type.

```julia
julia> num_constraints(model, GlobalVariableRef)
3
```
"""
function JuMP.num_constraints(model::InfiniteModel,
                            function_type::Type{<:Union{Symbol, JuMP.AbstractJuMPScalar}})::Int64
    return JuMP.num_constraints(model, function_type, MOI.AbstractSet)
end

"""
    JuMP.num_constraints(model::InfiniteModel,
                         function_type::Type{<:MOI.AbstractSet})::Int64

Extend [`JuMP.num_constraints`](@ref) to search by MOI set type for all function
types and return the total number of constraints that use a particular MOI set
type.

```julia
julia> num_constraints(model, MOI.LessThan)
2
```
"""
function JuMP.num_constraints(model::InfiniteModel,
                              set_type::Type{<:MOI.AbstractSet})::Int64
    return JuMP.num_constraints(model, JuMP.AbstractJuMPScalar, set_type)
end

"""
    JuMP.num_constraints(model::InfiniteModel)::Int64

Extend [`JuMP.num_constraints`](@ref) to return the total number of constraints
in an infinite model `model`.

```julia
julia> num_constraints(model)
4
```
"""
function JuMP.num_constraints(model::InfiniteModel)::Int64
    return length(model.constrs)
end

"""
    JuMP.all_constraints(model::InfiniteModel,
                         function_type::Type{<:JuMP.AbstractJuMPScalar},
                         set_type::Type{<:MOI.AbstractSet}
                         )::Vector{<:InfOptConstraintRef}

Extend [`JuMP.all_constraints`](@ref) to return a list of all the constraints
with a particular function type and set type.

```julia
julia> all_constraints(model, GlobalVariableRef, MOI.LessThan)
1-element Array{InfOptConstraintRef,1}:
 x <= 1.0
```
"""
function JuMP.all_constraints(model::InfiniteModel, function_type::Symbol,
                              set_type::Type{<:MOI.AbstractSet}
                              )::Vector{InfOptConstraintRef}
    constr_list = Vector{InfOptConstraintRef}(undef,
                           JuMP.num_constraints(model, function_type, set_type))
    indexes = sort(collect(keys(model.constrs)))
    counter = 1
    for index in indexes
        if isa(model.constrs[index].func, InfOptVariableRef) &&
           variable_type(model.constrs[index].func) == function_type
           isa(model.constrs[index].set, set_type)
            constr_list[counter] = _make_constraint_ref(model, index)
            counter += 1
        end
    end
    return constr_list
end

function JuMP.all_constraints(model::InfiniteModel,
                              function_type::Type{<:JuMP.AbstractJuMPScalar},
                              set_type::Type{<:MOI.AbstractSet}
                              )::Vector{InfOptConstraintRef}
    constr_list = Vector{InfOptConstraintRef}(undef,
                           JuMP.num_constraints(model, function_type, set_type))
    indexes = sort(collect(keys(model.constrs)))
    counter = 1
    for index in indexes
        if isa(model.constrs[index].func, function_type) && isa(model.constrs[index].set, set_type)
            constr_list[counter] = _make_constraint_ref(model, index)
            counter += 1
        end
    end
    return constr_list
end

"""
    JuMP.all_constraints(model::InfiniteModel,
                         function_type::Type{<:JuMP.AbstractJuMPScalar}
                         )::Vector{<:InfOptConstraintRef}

Extend [`JuMP.all_constraints`](@ref) to search by function types for all MOI
sets and return a list of all constraints use a particular function type.

```julia
julia> all_constraints(model, GlobalVariableRef)
3-element Array{InfOptConstraintRef,1}:
 x >= 0.0
 x <= 3.0
 x integer
```
"""
function JuMP.all_constraints(model::InfiniteModel,
                              function_type::Type{<:Union{Symbol, JuMP.AbstractJuMPScalar}}
                              )::Vector{<:InfOptConstraintRef}
    return JuMP.all_constraints(model, function_type, MOI.AbstractSet)
end

"""
    JuMP.all_constraints(model::InfiniteModel,
                         set_type::Type{<:MOI.AbstractSet}
                         )::Vector{<:InfOptConstraintRef}

Extend [`JuMP.all_constraints`](@ref) to search by MOI set type for all function
types and return a list of all constraints that use a particular set type.

```julia
julia> all_constraints(model, MOI.GreaterThan)
3-element Array{InfOptConstraintRef,1}:
 x >= 0.0
 g(t) >= 0.0
 g(0.5) >= 0.0
```
"""
function JuMP.all_constraints(model::InfiniteModel,
                              set_type::Type{<:MOI.AbstractSet}
                              )::Vector{<:InfOptConstraintRef}
    return JuMP.all_constraints(model, JuMP.AbstractJuMPScalar, set_type)
end

"""
    JuMP.all_constraints(model::InfiniteModel)::Vector{<:InfOptConstraintRef}

Extend [`JuMP.all_constraints`](@ref) to return all a list of all the constraints
in an infinite model `model`.

```julia
julia> all_constraints(model)
5-element Array{InfOptConstraintRef,1}:
 x >= 0.0
 x <= 3.0
 x integer
 g(t) >= 0.0
 g(0.5) >= 0.0
```
"""
function JuMP.all_constraints(model::InfiniteModel)::Vector{<:InfOptConstraintRef}
    return JuMP.all_constraints(model, JuMP.AbstractJuMPScalar, MOI.AbstractSet)
end

"""
    JuMP.list_of_constraint_types(model::InfiniteModel)::Vector{Tuple)

Extend [`JuMP.list_of_constraint_types`](@ref) to return a list of tuples that
contain all the used combinations of function types and set types in the model.

```julia
julia> all_constraints(model)
5-element Array{Tuple{Symbol,DataType},1}:
 (Global, MathOptInterface.LessThan{Float64})
 (Point, MathOptInterface.GreaterThan{Float64})
 (Global, MathOptInterface.GreaterThan{Float64})
 (Global, MathOptInterface.Integer)
 (Infinite, MathOptInterface.GreaterThan{Float64})
```
"""
function JuMP.list_of_constraint_types(model::InfiniteModel)::Vector{Tuple}
    type_list = Vector{Tuple{Symbol, DataType}}(undef,
                                                  JuMP.num_constraints(model))
    indexes = sort(collect(keys(model.constrs)))
    counter = 1
    for index in indexes
        type_list[counter] = (variable_type(model.constrs[index].func),
                              typeof(model.constrs[index].set))
        counter += 1
    end
    return unique(type_list)
end
