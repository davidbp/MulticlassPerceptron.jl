export MulticlassPerceptronClassifier

using DataFrames
using Tables
using CategoricalArrays

#import MLJMOdelInterface: target_scitype
import MLJModelInterface

# Add explicit case for table data not to use sparse format
import SparseArrays.issparse
issparse(X::DataFrame) = false  ## TODO: Add Tables option with sparse data when it is done


#= ##########################################################################
   Defining MLJMOdelInterface interface for a MulticlassPerceptronClassifier
########################################################################## =#

num_features_and_observations(X::DataFrame)     = reverse(size(X))  # (size(X,2), size(X,1))
num_features_and_observations(X::AbstractArray) = size(X)           # (size(X,1), size(X,2))


mutable struct MulticlassPerceptronClassifier <: MLJMOdelInterface.Deterministic
    n_epochs::Int
    n_epoch_patience::Int
    f_average_weights::Bool
    f_shuffle_data::Bool
    element_type::DataType
end

#target_scitype(::Type{MulticlassPerceptronClassifier}) = AbstractVector{<:MLJMOdelInterface.Finite}
descr(::Type{MulticlassPerceptronClassifier}) = "Classifier corresponding to a Multiclass Perceptron."


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

    message = MLJMOdelInterface.clean!(model)
    isempty(message) || @warn message
    return model
end

# should this be MLJ.clean! ?
function MLJMOdelInterface.clean!(model::MulticlassPerceptronClassifier)
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


function MLJModelInterface.fit(model::MulticlassPerceptronClassifier,
                     verbosity::Int,
                     X,
                     y)

    n_classes   = length(MLJMOdelInterface.classes(y[1]))

    if Tables.istable(X)
        X = MLJMOdelInterface.matrix(X, transpose=true)
    end

    n_features, _  = num_features_and_observations(X)

    decode  = MLJMOdelInterface.decoder(y[1]) # Storing a decode for the predict method
    y = Int.(MLJMOdelInterface.int(y))            # Encoding categorical target as array of integers

    is_sparse = issparse(X)
    perceptron = MulticlassPerceptronCore(model.element_type,
                                                    n_classes,
                                                    n_features,
                                                    is_sparse);

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


function MLJModelInterface.predict(model::MulticlassPerceptronClassifier, fitresult, Xnew)

    if Tables.istable(Xnew)
        Xnew = MLJBase.matrix(Xnew, transpose=true)
    end

    result, decode = fitresult
    prediction = _predict(result, Xnew)        # <------------ changed name to "private" name
    return decode(prediction)
end


function _predict(fitresult, Xnew)              # <----------- removed type annotation

    # Function fit!(MulticlassPerceptronCore, X, y) expects size(X) = n_features x n_observations
    if Xnew isa AbstractArray
        Xnew  = MLJBase.matrix(Xnew)
    elseif Tables.istable(Xnew)
        Xnew  = MLJBase.matrix(Xnew, transpose=true)
    end

    result, decode = fitresult
    prediction     = predict(result, Xnew)
    return decode(prediction)
end


#= =======================
   METADATA FOR ALL MODELS
   ======================= =#

descr_(M) = descr(M) *
    "\n→ based on [MulticlassPerceptron](https://github.com/davidbp/MulticlassPerceptron.jl)" *
    "\n→ do `@load $(MLJMOdelInterface.name(M)) pkg=\"MulticlassPerceptron\" to use the model.`" *
    "\n→ do `?$(MLJMOdelInterface.name(M))` for documentation."

lp_(M) = "MulticlassPerceptron.$(MLJMOdelInterface.name(M))"


MLJMOdelInterface.metadata_model(MulticlassPerceptronClassifier,
    input=MLJMOdelInterface.Table(MLJMOdelInterface.Continuous),
    target=AbstractVector{<:MLJMOdelInterface.Finite},
    weights=false,
    descr=descr_(MulticlassPerceptronClassifier),
    path=lp_(MulticlassPerceptronClassifier))

