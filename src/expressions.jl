#=
## Define type hierchical parser for use in building expressions with mixed types
## This is tested in test/operators.jl
# MeasureRef
function _var_type_parser(V::T, W::T) where {T<:Type{<:MeasureRef}}
    return MeasureRef
end

# ParameterRef
function _var_type_parser(V::T, W::T) where {T<:Type{<:ParameterRef}}
    return ParameterRef
end

#=
# InfiniteVariableRef
function _var_type_parser(V::T, W::T) where {T<:Type{<:InfiniteVariableRef}}
    return InfiniteVariableRef
end

# PointVariableRef
function _var_type_parser(V::T, W::T) where {T<:Type{<:PointVariableRef}}
    return PointVariableRef
end

# GlobalVariableRef
function _var_type_parser(V::T, W::T) where {T<:Type{<:GlobalVariableRef}}
    return GlobalVariableRef
end
=#

# InfOptVariableRef
function _var_type_parser(V::T, W::T) where {T<:Type{<:InfOptVariableRef}}
    return InfOptVariableRef
end

# ReducedInfiniteVariableRef
function _var_type_parser(V::T, W::T) where {T<:Type{<:ReducedInfiniteVariableRef}}
    return ReducedInfiniteVariableRef
end

#=
# FiniteVariableRef
function _var_type_parser(V::Type{<:FiniteVariableRef},
                          W::Type{<:FiniteVariableRef})
    return FiniteVariableRef
end
=#

# MeasureFiniteVariableRef
function _var_type_parser(V::Type{<:MeasureFiniteVariableRef},
                          W::Type{<:MeasureFiniteVariableRef})
    return MeasureFiniteVariableRef
end

# GeneralVariableRef
function _var_type_parser(V::Type{<:GeneralVariableRef},
                          W::Type{<:GeneralVariableRef})
    return GeneralVariableRef
end
=#
## Extend add_to_expression! for some more functionality, tested in test/operators.jl
# Mixed variable addition
function JuMP.add_to_expression!(quad::JuMP.GenericQuadExpr{C, InfOptVariableRef},
                                 new_coef::C,
                                 new_var1::InfOptVariableRef,
                                 new_var2::InfOptVariableRef
                                 )::JuMP.GenericQuadExpr where {C}
    key = JuMP.UnorderedPair{InfOptVariableRef}(new_var1, new_var2)
    new_quad = convert(JuMP.GenericQuadExpr{C, InfOptVariableRef}, quad)
    JuMP._add_or_set!(new_quad.terms, key, new_coef)
    return new_quad
end

# var1 is a number
function JuMP.add_to_expression!(quad::JuMP.GenericQuadExpr{C, InfOptVariableRef},
                                 new_coef::Number, new_var1::Number,
                                 new_var2::InfOptVariableRef
                                 )::JuMP.GenericQuadExpr where {C}
    new_quad = convert(JuMP.GenericQuadExpr{C, InfOptVariableRef}, quad)
    return JuMP.add_to_expression!(new_quad, new_coef * new_var1, new_var2)
end

# var2 is a number
function JuMP.add_to_expression!(quad::JuMP.GenericQuadExpr{C, InfOptVariableRef},
                                 new_coef::Number, new_var1::InfOptVariableRef,
                                 new_var2::Number
                                 )::JuMP.GenericQuadExpr where {C}
    new_quad = convert(JuMP.GenericQuadExpr{C, InfOptVariableRef}, quad)
    return JuMP.add_to_expression!(new_quad, new_coef * new_var2, new_var1)
end

# var1 and var2 are numbers
function JuMP.add_to_expression!(quad::JuMP.GenericQuadExpr, new_coef::Number,
                                 new_var1::Number, new_var2::Number
                                 )::JuMP.GenericQuadExpr
    JuMP.add_to_expression!(quad.aff, new_coef * new_var2 * new_var1)
    return quad
end

## Extend destructive_add! as needed
# Variable, constant, variable
function JuMP.destructive_add!(ex::InfOptVariableRef, c::InfOptVariableRef,
                               x::InfOptVariableRef)::JuMP.GenericAffExpr
#    type = _var_type_parser(Z, V)
    return JuMP.GenericAffExpr{InfOptVariableRef, type}(0.0, ex => 1.0, x => JuMP._float(c))
end

## Delete everything above here

## Extend for better comparisons than default
# GenericAffExpr
function Base.:(==)(aff1::JuMP.GenericAffExpr{InfOptVariableRef, InfOptVariableRef},
                    aff2::JuMP.GenericAffExpr{InfOptVariableRef, InfOptVariableRef})
    return aff1.constant == aff2.constant && collect(pairs(aff1.terms)) == collect(pairs(aff2.terms))
end

# GenericQuadExpr
function Base.:(==)(quad1::JuMP.GenericQuadExpr{InfOptVariableRef, InfOptVariableRef},
                    quad2::JuMP.GenericQuadExpr{InfOptVariableRef, InfOptVariableRef})
    pairs1 = collect(pairs(quad1.terms))
    pairs2 = collect(pairs(quad2.terms))
    if length(pairs1) != length(pairs2)
        return false
    end
    for i in eachindex(pairs1)
        if pairs1[i][1].a != pairs2[i][1].a || pairs1[i][1].b != pairs2[i][1].b || pairs1[i][2] != pairs2[i][2]
            return false
        end
    end
    return quad1.aff == quad2.aff
end

## Determine which variables are present in a function
# InfOptVariableRef
function _all_function_variables(f::InfOptVariableRef)::Vector{InfOptVariableRef}
    return [f]
end

# GenericAffExpr
function _all_function_variables(f::JuMP.GenericAffExpr)::Vector{InfOptVariableRef}
    return InfOptVariableRef[vref for vref in keys(f.terms)]
end

# GenericQuadExpr
function _all_function_variables(f::JuMP.GenericQuadExpr)::Vector{InfOptVariableRef}
    aff_vrefs = _all_function_variables(f.aff)
    vref_pairs = [k for k in keys(f.terms)]
    a_vrefs = InfOptVariableRef[pair.a for pair in vref_pairs]
    b_vrefs = InfOptVariableRef[pair.b for pair in vref_pairs]
    return unique([aff_vrefs; a_vrefs; b_vrefs])
end

# Fallback
function _all_function_variables(f)
    error("Can only use InfiniteOpt variables and expressions.")
    return
end

## Return a tuple of the parameter references in an expr
# Function wrapper for _all_parameter_refs
function _all_parameter_refs(expr::InfOptVariableRef)::Tuple
    return _all_parameter_refs(expr::InfOptVariableRef, Val(variable_type(expr)))
end

# Finite variable reference
_all_parameter_refs(expr::InfOptVariableRef, ::Union{Val{Point}, Val{Global}})::Tuple = ()

# Infinite variable references
_all_parameter_refs(expr::InfOptVariableRef, ::Val{Infinite})::Tuple = parameter_refs(expr)

# Parameter references
_all_parameter_refs(expr::InfOptVariableRef, ::Val{Parameter})::Tuple = (expr, )

# Reduced infinite variable references
_all_parameter_refs(expr::InfOptVariableRef, ::Val{Reduced})::Tuple = parameter_refs(expr)

# GenericAffExpr
function _all_parameter_refs(expr::JuMP.GenericAffExpr{C,
                             InfOptVariableRef})::Tuple where {C}
    pref_list = []
    for var in keys(expr.terms)
        push!(pref_list, _all_parameter_refs(var)...)
    end
    groups = _group.(pref_list)
    unique_groups = unique(groups)
    return Tuple(pref_list[findfirst(isequal(unique_groups[i]), groups)]
                 for i in eachindex(unique_groups))
end

# GenericQuadExpr
function _all_parameter_refs(expr::JuMP.GenericQuadExpr{C,
                             InfOptVariableRef})::Tuple where {C}
    pref_list = Any[i for i in _all_parameter_refs(expr.aff)]
    for pair in keys(expr.terms)
        push!(pref_list, _all_parameter_refs(pair.a)...)
        push!(pref_list, _all_parameter_refs(pair.b)...)
    end
    groups = _group.(pref_list)
    unique_groups = unique(groups)
    return Tuple(pref_list[findfirst(isequal(unique_groups[i]), groups)]
                 for i in eachindex(unique_groups))
end

## Delete variables from an expression
# GenericAffExpr
function _remove_variable(f::JuMP.GenericAffExpr, vref::InfOptVariableRef)
    if haskey(f.terms, vref)
        delete!(f.terms, vref)
    end
    return
end

# GenericQuadExpr
function _remove_variable(f::JuMP.GenericQuadExpr, vref::InfOptVariableRef)
    _remove_variable(f.aff, vref)
    vref_pairs = [k for k in keys(f.terms)]
    for i = 1:length(vref_pairs)
        if vref_pairs[i].a == vref
            delete!(f.terms, vref_pairs[i])
        elseif vref_pairs[i].b == vref
            delete!(f.terms, vref_pairs[i])
        end
    end
    return
end

## Modify linear coefficient of variable in expression
# GeneralVariableRef
function _set_variable_coefficient!(expr::InfOptVariableRef,
                                    var::InfOptVariableRef,
                                    coeff::Real)::JuMP.GenericAffExpr
    # Determine if variable is that of the expression and change accordingly
    if expr == var
        return coeff * var
    else
        return expr + coeff * var
    end
end

# GenericAffExpr
function _set_variable_coefficient!(expr::JuMP.GenericAffExpr,
                                    var::InfOptVariableRef,
                                    coeff::Real)::JuMP.GenericAffExpr
    # Determine if variable is in the expression and change accordingly
    if haskey(expr.terms, var)
        expr.terms[var] = coeff
        return expr
    else
        return expr + coeff * var
    end
end

# GenericQuadExpr
function _set_variable_coefficient!(expr::JuMP.GenericQuadExpr,
                                    var::InfOptVariableRef,
                                    coeff::Real)::JuMP.GenericQuadExpr
    # Determine if variable is in the expression and change accordingly
    if haskey(expr.aff.terms, var)
        expr.aff.terms[var] = coeff
        return expr
    else
        return expr + coeff * var
    end
end

# Fallback
function _set_variable_coefficient!(expr, var::InfOptVariableRef, coeff::Real)
    error("Unsupported function type for coefficient modification.")
end

# Check expression for a particular variable type via a recursive search
# This is tested in test/measures.jl
function _has_variable(vrefs::Vector{InfOptVariableRef},
                       vref::InfOptVariableRef; prior=[])
    if vrefs[1] == vref
        return true
    elseif variable_type(vrefs[1]) == MeasureRef
        if length(vrefs) > 1
            return _has_variable(_all_function_variables(measure_function(vrefs[1])),
                          vref, prior = InfOptVariableRef[prior; vrefs[2:end]])
        else
            return _has_variable(_all_function_variables(measure_function(vrefs[1])),
                                 vref, prior = prior)
        end
    elseif length(vrefs) > 1
        return _has_variable(vrefs[2:end], vref, prior = prior)
    elseif length(prior) > 0
        return _has_variable(prior, vref)
    else
        return false
    end
end
