"""
    InfiniteOpt.map_value(vref::InfiniteOpt.InfOptVariableRef,
                          key::Val{:TransData})

Map the value of the appropriate transcription variable in the transcription
model to `vref`.
"""
function InfiniteOpt.map_value(vref::InfiniteOpt.InfOptVariableRef,
                               key::Val{:TransData})
    trans_model = InfiniteOpt.optimizer_model(JuMP.owner_model(vref))
    return JuMP.value.(transcription_variable(trans_model, vref))
end

"""
    InfiniteOpt.map_value(vref::InfiniteOpt.InfOptConstraintRef,
                          key::Val{:TransData})

Map the value of the appropriate transcription constraint function in the
transcription model to `cref`.
"""
function InfiniteOpt.map_value(icref::InfiniteOpt.InfOptConstraintRef,
                               key::Val{:TransData})
    trans_model = InfiniteOpt.optimizer_model(JuMP.owner_model(icref))
    return JuMP.value.(transcription_constraint(trans_model, icref))
end

"""
    InfiniteOpt.map_optimizer_index(vref::InfiniteOpt.InfOptVariableRef,
                                    key::Val{:TransData})

Map the optimizer model index of the appropriate transcription variable in the
transcription model to `vref`.
"""
function InfiniteOpt.map_optimizer_index(vref::InfiniteOpt.InfOptVariableRef,
                                         key::Val{:TransData})
    trans_model = InfiniteOpt.optimizer_model(JuMP.owner_model(vref))
    return JuMP.optimizer_index.(transcription_variable(trans_model, vref))
end

"""
    InfiniteOpt.map_optimizer_index(cref::InfiniteOpt.InfOptConstraintRef,
                                    key::Val{:TransData})

Map the optimizer model index of the appropriate transcription constraints in the
transcription model to `cref`.
"""
function InfiniteOpt.map_optimizer_index(cref::InfiniteOpt.InfOptConstraintRef,
                                         key::Val{:TransData})
    trans_model = InfiniteOpt.optimizer_model(JuMP.owner_model(cref))
    return JuMP.optimizer_index.(transcription_constraint(trans_model, cref))
end

"""
    InfiniteOpt.map_dual(cref::InfiniteOpt.InfOptConstraintRef,
                         key::Val{:TransData})

Map the duals of the appropriate transcription constraints in the
transcription model to `cref`.
"""
function InfiniteOpt.map_dual(ccref::InfiniteOpt.InfOptConstraintRef,
                              key::Val{:TransData})
    trans_model = InfiniteOpt.optimizer_model(JuMP.owner_model(ccref))
    return JuMP.dual.(transcription_constraint(trans_model, ccref))
end

"""
    InfiniteOpt.map_shadow_price(icref::InfiniteOpt.InfOptConstraintRef,
                                 key::Val{:TransData})

Map the shadow prices of the appropriate transcription constraints in the
transcription model to `icref`.
"""
function InfiniteOpt.map_shadow_price(icref::InfiniteOpt.InfOptConstraintRef,
                                      key::Val{:TransData})
    trans_model = InfiniteOpt.optimizer_model(JuMP.owner_model(icref))
    return JuMP.shadow_price.(transcription_constraint(trans_model, icref))
end
