export MulticlassPerceptronClassifier

using DataFrames
using Tables
using CategoricalArrays

import MLJ
import MLJBase

# Add explicit case for table data not to use sparse format
import SparseArrays.issparse
issparse(X::DataFrame) = false  ## TODO: Add Tables option with sparse data when it is done


#= ##########################################################################
   Defining MLJ interface for a MulticlassPerceptronClassifier
########################################################################## =#

num_features_and_observations(X::DataFrame)     = (size(X,2), size(X,1))
num_features_and_observations(X::AbstractArray) = (size(X,1), size(X,2))

mutable struct MulticlassPerceptronClassifier <: MLJBase.Deterministic
    n_epochs::Int
    n_epoch_patience::Int
    f_average_weights::Bool
    f_shuffle_data::Bool
    element_type::DataType
end


const CLF_MODELS = (MulticlassPerceptronClassifier)
const ALL_MODELS = (MulticlassPerceptronClassifier)


# keyword constructor
function MulticlassPerceptronClassifier( ;
                                        n_epochs=100,
                                        n_epoch_patience=5,
                                        f_average_weights=true,
                                        f_shuffle_data=false,
                                        element_type=Float32)

    model = MulticlassPerceptronClassifier(n_epochs,
                                           n_epoch_patience,
                                           f_average_weights,
                                           f_shuffle_data,
                                           element_type)

    message = MLJBase.clean!(model)
    isempty(message) || @warn message
    return model
end


function MLJ.clean!(model::MulticlassPerceptronClassifier)
    warning = ""
    if model.n_epochs < 1
        warning *= "Need n_epochs ≥ 1. Resetting n_epochs=100 "
        model.n_epochs = 50
    end

    if model.n_epoch_patience <1
        warning *= "Need epoch_patience ≥ 1. Resetting epoch_patience=5 "
        model.epoch_patience = 5
    end

    return warning
end


function MLJBase.predict(model::MulticlassPerceptronClassifier, fitresult, Xnew)
    
    # Function fit!(perceptron, X, y) expects size(X) = n_features x n_observations
    if Xnew isa AbstractArray
        Xnew  = MLJBase.matrix(Xnew)
    elseif Tables.istable(Xnew) 
        Xnew  = MLJBase.matrix(Xnew, transpose=true)
    end

    result, decode = fitresult
    prediction = predict(result, Xnew)
    return decode(prediction)
end



function MLJBase.fit(model::MulticlassPerceptronClassifier,
                     verbosity::Int,
                     X,
                     y::CategoricalArray)

    
    n_classes   = length(MLJ.classes(y[1]))

    if X isa AbstractArray
        X  = MLJBase.matrix(X)
    elseif  Tables.istable(X) 
        X  = MLJBase.matrix(X, transpose=true)
    end

    n_features, _  = num_features_and_observations(X)

    decode  = MLJBase.decoder(y[1]) # for the predict method
    y = Int.(MLJ.int(y))            # Encoding categorical target as array of integers

    is_sparse = issparse(X)
    perceptron = MulticlassPerceptronClassifierCore(model.element_type, 
                                                    n_classes,
                                                    n_features, 
                                                    is_sparse);
    println("training")

    ### Fitting code starts
    fit!(perceptron, X, y;
         verbosity=verbosity,
         n_epochs=model.n_epochs,
         f_average_weights=model.f_average_weights,
         f_shuffle_data=model.f_shuffle_data
        );

    ### Fitting code ends
    cache = nothing
    fitresult = (perceptron, decode)
    report = NamedTuple{}()

    return fitresult, cache, report
end


function MLJBase.predict(fitresult::Tuple{MulticlassPerceptronClassifierCore, MLJBase.CategoricalDecoder}, Xnew)

    # Function fit!(MulticlassPerceptronClassifierCore, X, y) expects size(X) = n_features x n_observations
    if Xnew isa AbstractArray
        Xnew  = MLJBase.matrix(Xnew)
    elseif Tables.istable(Xnew) 
        Xnew  = MLJBase.matrix(Xnew, transpose=true)
    end

    result, decode = fitresult
    prediction     = predict(result, Xnew)
    return decode(prediction)
end



