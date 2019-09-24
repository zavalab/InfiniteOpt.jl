"""
    TranscriptionData

A DataType for storing the data mapping an `InfiniteModel` that has been
transcribed to a regular JuMP `Model` that contains the transcribed variables.

**Fields**
- `infinite_to_vars::Dict{InfiniteOpt.InfOptVariableRef,
   Vector{JuMP.VariableRef}}`: Infinite variables to their transcribed variables.
- `global_to_var::Dict{InfiniteOpt.InfOptVariableRef, JuMP.VariableRef}`: Global
  variables to model variables.
- `point_to_var::Dict{InfiniteOpt.InfOptVariableRef, JuMP.VariableRef}`: Point
  variables to model variables.
- `infvar_to_supports::Dict{InfiniteOpt.InfOptVariableRef, Dict}`: Infinite
  variables to transcribed supports indexed by their numeric aliases.
- `infinite_to_constrs::Dict{InfiniteOpt.InfOptConstraintRef,
  Vector{JuMP.ConstraintRef}}`: Infinite constraints to their transcribed
  constraints.
- `measure_to_constrs::Dict{InfiniteOpt.InfOptConstraintRef,
  Vector{JuMP.ConstraintRef}}`: Measure constraints to model constraints.
- `finite_to_constr::Dict{InfiniteOpt.InfOptConstraintRef, JuMP.ConstraintRef}`:
  Finite constraints to model constraints.
- `infconstr_to_supports::Dict{InfiniteOpt.InfOptConstraintRef, Dict}`: Infinite
  constraints to the transcribed supports indxed by their numeric aliases.
- `measconstr_to_supports::Dict{InfiniteOpt.InfOptConstraintRef, Dict}`:
  Measure constraints to the transcribed supports indxed by their numeric aliases.
- `infconstr_to_params::Dict{InfiniteOpt.InfOptConstraintRef, Tuple}`: Infinite
  constraints to the parameter tuples associated with each transcribed support.
- `measconstr_to_params::Dict{InfiniteOpt.InfOptConstraintRef, Tuple}`: Measure
  constraints to the parameter tuples associated with each transcribed support.
"""
mutable struct TranscriptionData
    # Variable mapping
    infinite_to_vars::Dict{InfiniteOpt.InfOptVariableRef,
                           Vector{JuMP.VariableRef}}
    global_to_var::Dict{InfiniteOpt.InfOptVariableRef, JuMP.VariableRef}
    point_to_var::Dict{InfiniteOpt.InfOptVariableRef, JuMP.VariableRef}

    # Variable support data
    infvar_to_supports::Dict{InfiniteOpt.InfOptVariableRef, Vector{<:Tuple}}

    # Constraint mapping
    infinite_to_constrs::Dict{InfiniteOpt.InfOptConstraintRef,
                             Vector{JuMP.ConstraintRef}}
    measure_to_constrs::Dict{InfiniteOpt.InfOptConstraintRef,
                            Vector{JuMP.ConstraintRef}}
    finite_to_constr::Dict{InfiniteOpt.InfOptConstraintRef, JuMP.ConstraintRef}

    # Constraint support data
    infconstr_to_supports::Dict{InfiniteOpt.InfOptConstraintRef, Vector{<:Tuple}}
    measconstr_to_supports::Dict{InfiniteOpt.InfOptConstraintRef, Vector{<:Tuple}}
    infconstr_to_params::Dict{InfiniteOpt.InfOptConstraintRef, Tuple}
    measconstr_to_params::Dict{InfiniteOpt.InfOptConstraintRef, Tuple}

    # Default constructor
    function TranscriptionData()
        return new(Dict{InfiniteOpt.InfOptVariableRef,
                   Vector{JuMP.VariableRef}}(),
                   Dict{InfiniteOpt.InfOptVariableRef, JuMP.VariableRef}(),
                   Dict{InfiniteOpt.InfOptVariableRef, JuMP.VariableRef}(),
                   Dict{InfiniteOpt.InfOptVariableRef, Vector{Tuple}}(),
                   Dict{InfiniteOpt.InfOptConstraintRef,
                        Vector{JuMP.ConstraintRef}}(),
                   Dict{InfiniteOpt.InfOptConstraintRef,
                        Vector{JuMP.ConstraintRef}}(),
                   Dict{InfiniteOpt.InfOptConstraintRef, JuMP.ConstraintRef}(),
                   Dict{InfiniteOpt.InfOptConstraintRef, Vector{Tuple}}(),
                   Dict{InfiniteOpt.InfOptConstraintRef, Vector{Tuple}}(),
                   Dict{InfiniteOpt.InfOptConstraintRef, Tuple}(),
                   Dict{InfiniteOpt.InfOptConstraintRef, Tuple}())
    end
end

"""
    TranscriptionModel(args...)::JuMP.Model

Return a JuMP `Model` with `TranscriptionData` included in the extension
data field. Accepts the same arguments as a typical JuMP `Model`.

**Example**
```julia
julia> TranscriptionModel()
A JuMP Model
Feasibility problem with:
Variables: 0
Model mode: AUTOMATIC
CachingOptimizer state: NO_OPTIMIZER
Solver name: No optimizer attached.
```
"""
function TranscriptionModel(; kwargs...)::JuMP.Model
    model = JuMP.Model(; kwargs...)
    model.ext[:TransData] = TranscriptionData()
    return model
end
# Accept optimizer_factorys
function TranscriptionModel(optimizer_factory::JuMP.OptimizerFactory;
                            kwargs...)::JuMP.Model
    model = JuMP.Model(optimizer_factory; kwargs...)
    model.ext[:TransData] = TranscriptionData()
    return model
end

"""
    is_transcription_model(model::JuMP.Model)::Bool

Return true if `model` is a `TranscriptionModel` or false otherwise.

**Example**
```julia
julia> is_transcription_model(model)
true
```
"""
function is_transcription_model(model::JuMP.Model)::Bool
    return haskey(model.ext, :TransData)
end

"""
    transcription_data(model::JuMP.Model)::TranscriptionData

Return the `TranscriptionData` from a `TranscriptionModel`. Errors if it is not
a `TranscriptionModel`.
"""
function transcription_data(model::JuMP.Model)::TranscriptionData
    !is_transcription_model(model) && error("Model is not a transcription model.")
    return model.ext[:TransData]
end

"""
    transcription_variable(model::JuMP.Model,
                           vref::InfiniteOpt.InfOptVariableRef)

Return the transcribed variable reference(s) corresponding to `vref`. Errors
if no transcription variable is found.

**Example**
```julia
julia> transcription_variable(trans_model, infvar)
2-element Array{VariableRef,1}:
 infvar(support: 1)
 infvar(support: 2)

julia> transcription_variable(trans_model, gbvar)
gbvar
```
"""
function transcription_variable end

## Define the variable mapping functions
# function wrapper for transcription_variable
function transcription_variable(model::JuMP.Model,
                                vref::InfiniteOpt.InfOptVariableRef
                                )::Union{JuMP.VariableRef, Vector}
    return transcription_variable(model, vref, Val(InfiniteOpt.variable_type(vref)))
end

# global Variable refs
function transcription_variable(model::JuMP.Model,
                                vref::InfiniteOpt.InfOptVariableRef,
                                ::Val{InfiniteOpt.Global})::JuMP.VariableRef
    !haskey(transcription_data(model).global_to_var, vref) && error("Variable " *
                             "reference $vref not used in transcription model.")
    return transcription_data(model).global_to_var[vref]
end
# Infinite variable refs
function transcription_variable(model::JuMP.Model,
                                vref::InfiniteOpt.InfOptVariableRef,
                                ::Val{InfiniteOpt.Infinite})::Vector
    !haskey(transcription_data(model).infinite_to_vars, vref) && error("Variable" *
                             "reference $vref not used in transcription model.")
    return transcription_data(model).infinite_to_vars[vref]
end
# Point variable refs
function transcription_variable(model::JuMP.Model,
                                vref::InfiniteOpt.InfOptVariableRef,
                                ::Val{InfiniteOpt.Point})::JuMP.VariableRef
    !haskey(transcription_data(model).point_to_var, vref) && error("Variable " *
                             "reference $vref not used in transcription model.")
    return transcription_data(model).point_to_var[vref]
end

"""
    InfiniteOpt.supports(model::JuMP.Model,
                         vref::InfiniteOpt.InfOptVariableRef)::Vector

Return the support alias mapping associated with `vref` in the transcribed model.
Errors if `vref` does not have transcribed variables.
"""
function InfiniteOpt.supports(model::JuMP.Model,
                              vref::InfiniteOpt.InfOptVariableRef)::Vector
    return InfiniteOpt.supports(model, vref,
                                Val(InfiniteOpt.variable_type(vref)))
end

function InfiniteOpt.supports(model::JuMP.Model,
                              vref::InfiniteOpt.InfOptVariableRef,
                              ::Val{InfiniteOpt.Infinite})::Vector
    if !haskey(transcription_data(model).infvar_to_supports, vref)
        error("Variable reference $vref not used in transcription model.")
    end
    return transcription_data(model).infvar_to_supports[vref]
end

"""
    InfiniteOpt.supports(vref::InfiniteOpt.InfOptVariableRef)::Vector

Return the support alias mapping associated with `vref` in the transcription
model. Errors if the infinite model does not contain a transcription model or if
`vref` is not transcribed.

**Example**
```julia
julia> supports(vref)
Dict{Int64,Tuple{Float64}} with 2 entries:
  2 => (1.0,)
  1 => (0.0,)
```
"""
function InfiniteOpt.supports(vref::InfiniteOpt.InfOptVariableRef)::Vector
    return InfiniteOpt.supports(vref, Val(InfiniteOpt.variable_type(vref)))
end

function InfiniteOpt.supports(vref::InfiniteOpt.InfOptVariableRef,
                              ::Val{InfiniteOpt.Infinite})::Vector
    model = InfiniteOpt.optimizer_model(JuMP.owner_model(vref))
    return InfiniteOpt.supports(model, vref)
end

"""
    transcription_constraint(model::JuMP.Model,
                             cref::InfiniteOpt.InfOptConstraintRef)

Return the transcribed constraint reference(s) corresponding to `cref`. Errors
if `cref` has not been transcribed.

**Example**
```julia
julia> transcription_constraint(trans_model, fin_con)
fin_con : x(support: 1) - y <= 3.0
```
"""

## Define the cosntraint mapping functions
# function wrapper for transcription_constraint
function transcription_constraint(model::JuMP.Model,
                                  cref::InfiniteOpt.InfOptConstraintRef
                                  )::Union{JuMP.ConstraintRef, Vector}
    return transcription_constraint(model, cref, Val(InfiniteOpt.constraint_type(cref)))
end

# Infinite constraint refs
function transcription_constraint(model::JuMP.Model,
                                  cref::InfiniteOpt.InfOptConstraintRef,
                                  ::Val{InfiniteOpt.Infinite})::Vector
    if !haskey(transcription_data(model).infinite_to_constrs, cref)
        error("Constraint reference $cref not used in transcription model.")
    end
    return transcription_data(model).infinite_to_constrs[cref]
end

# Measure Constraint refs
function transcription_constraint(model::JuMP.Model,
                                  cref::InfiniteOpt.InfOptConstraintRef,
                                  ::Val{InfiniteOpt.MeasureRef})::Vector
    if !haskey(transcription_data(model).measure_to_constrs, cref)
        error("Constraint reference $cref not used in transcription model.")
    end
    return transcription_data(model).measure_to_constrs[cref]
end

# Finite constraint refs
function transcription_constraint(model::JuMP.Model,
                                  cref::InfiniteOpt.InfOptConstraintRef,
                                  ::Val{InfiniteOpt.Finite})::JuMP.ConstraintRef
    if !haskey(transcription_data(model).finite_to_constr, cref)
        error("Constraint reference $cref not used in transcription model.")
    end
    return transcription_data(model).finite_to_constr[cref]
end

"""
    InfiniteOpt.supports(model::JuMP.Model,
                         cref::InfiniteOpt.InfOptConstraintRef)::Vector

Return the support alias mappings associated with `cref`. Errors if `cref` is
not transcribed.
"""
# function wrapper for InfiniteOpt.supports for constraint refs
function InfiniteOpt.supports(model::JuMP.Model,
                              cref::InfiniteOpt.InfOptConstraintRef)::Vector
    return InfiniteOpt.supports(model, cref, Val(InfiniteOpt.constraint_type(cref)))
end
# Infinite constraint refs
function InfiniteOpt.supports(model::JuMP.Model,
                              cref::InfiniteOpt.InfOptConstraintRef,
                              ::Val{InfiniteOpt.Infinite})::Vector
    if !haskey(transcription_data(model).infconstr_to_supports, cref)
        error("Constraint reference $cref not used in transcription model.")
    end
    return transcription_data(model).infconstr_to_supports[cref]
end
# Measure constraint refs
function InfiniteOpt.supports(model::JuMP.Model,
                              cref::InfiniteOpt.InfOptConstraintRef,
                              ::Val{InfiniteOpt.MeasureRef})::Vector
    if !haskey(transcription_data(model).measconstr_to_supports, cref)
        error("Constraint reference $cref not used in transcription model " *
              "and/or is finite and doesn't have supports.")
    end
    return transcription_data(model).measconstr_to_supports[cref]
end

"""
    InfiniteOpt.supports(cref::InfiniteOpt.InfOptConstraintRef)::Vector

Return the support alias mappings associated with `cref`. Errors if `cref` is
not transcribed or if the infinite model does not have a transcription model.

**Example**
```julia
julia> supports(cref)
Dict{Int64,Tuple{Float64}} with 2 entries:
  2 => (1.0,)
  1 => (0.0,)
```
"""
# function wrapper
function InfiniteOpt.supports(cref::InfiniteOpt.InfOptConstraintRef)::Vector
    return InfiniteOpt.supports(cref, Val(InfiniteOpt.constraint_type(cref)))
end
# Infinite constraint refs
function InfiniteOpt.supports(cref::InfiniteOpt.InfOptConstraintRef,
                              ::Val{InfiniteOpt.Infinite})::Vector
    model = InfiniteOpt.optimizer_model(JuMP.owner_model(cref))
    return InfiniteOpt.supports(model, cref)
end
# Measure constraint refs
function InfiniteOpt.supports(cref::InfiniteOpt.InfOptConstraintRef,
                              ::Val{InfiniteOpt.MeasureRef})::Vector
    model = InfiniteOpt.optimizer_model(JuMP.owner_model(cref))
    return InfiniteOpt.supports(model, cref)
end

"""
    InfiniteOpt.parameter_refs(model::JuMP.Model,
                               cref::InfiniteOpt.InfOptConstraintRef)::Tuple

Return the a parameter reference tuple of all the parameters that parameterize
`cref` and correspond to the supports. Errors if `cref` has not been transcribed.
"""
function InfiniteOpt.parameter_refs(model::JuMP.Model,
                                    cref::InfiniteOpt.InfOptConstraintRef)::Tuple
    return InfiniteOpt.parameter_refs(model, cref, Val(InfiniteOpt.constraint_type(cref)))
end
# Infinite constraint refs
function InfiniteOpt.parameter_refs(model::JuMP.Model,
                                    cref::InfiniteOpt.InfOptConstraintRef,
                                    ::Val{InfiniteOpt.Infinite})::Tuple
    if !haskey(transcription_data(model).infconstr_to_params, cref)
        error("Constraint reference $cref not used in transcription model " *
              "and/or is finite and doesn't have parameter references.")
    end
    return transcription_data(model).infconstr_to_params[cref]
end
# Measure constraint ref
function InfiniteOpt.parameter_refs(model::JuMP.Model,
                                    cref::InfiniteOpt.InfOptConstraintRef,
                                    ::Val{InfiniteOpt.MeasureRef})::Tuple
    if !haskey(transcription_data(model).measconstr_to_params, cref)
        error("Constraint reference $cref not used in transcription model " *
              "and/or is finite and doesn't have parameter references.")
    end
    return transcription_data(model).measconstr_to_params[cref]
end

"""
    InfiniteOpt.parameter_refs(cref::InfiniteOpt.InfOptConstraintRef)::Tuple

Return the a parameter reference tuple of all the parameters that parameterize
`cref` and correspond to the supports. Errors if `cref` has not been transcribed
or if the infinite model does not have a transcription model associated with it.

**Example**
```julia
julia> parameter_refs(cref)
(t, x)
```
"""
# function wrapper
function InfiniteOpt.parameter_refs(cref::InfiniteOpt.InfOptConstraintRef)::Tuple
    return InfiniteOpt.parameter_refs(cref, Val(InfiniteOpt.constraint_type(cref)))
end
# Infinite constraint refs
function InfiniteOpt.parameter_refs(cref::InfiniteOpt.InfOptConstraintRef,
                                    ::Val{InfiniteOpt.Infinite})::Tuple
    model = InfiniteOpt.optimizer_model(JuMP.owner_model(cref))
    return InfiniteOpt.parameter_refs(model, cref)
end
# Measure constraint refs
function InfiniteOpt.parameter_refs(cref::InfiniteOpt.InfOptConstraintRef,
                                    ::Val{InfiniteOpt.MeasureRef})::Tuple
    model = InfiniteOpt.optimizer_model(JuMP.owner_model(cref))
    return InfiniteOpt.parameter_refs(model, cref)
end
