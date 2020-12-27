export MulticlassPerceptronClassifier

using Tables
using CategoricalArrays
import MLJModelInterface

# Add explicit case for table data not to use sparse format
import SparseArrays.issparse

#= ##########################################################################
   Defining MLJModelInterface interface for a MulticlassPerceptronClassifier
########################################################################## =#

mutable struct MulticlassPerceptronClassifier <: MLJModelInterface.Deterministic
    n_epochs::Int
    n_epoch_patience::Int
    f_average_weights::Bool
    f_shuffle_data::Bool
    element_type::DataType
end

descr(::Type{MulticlassPerceptronClassifier}) =
    "Classifier corresponding to a Multiclass Perceptron."

const CLF_MODELS = (MulticlassPerceptronClassifier)
const ALL_MODELS = (MulticlassPerceptronClassifier)

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

    message = MLJModelInterface.clean!(model)
    isempty(message) || @warn message
    return model
end

function MLJModelInterface.clean!(model::MulticlassPerceptronClassifier)
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

# front end for incoming features, which can only be tables or
# matrices; the output of these methods is an abstract array with
# features as rows:
_reformat(X) = MLJModelInterface.matrix(X, transpose=true) # fallback is table
_reformat(X, ::Type{<:AbstractMatrix}) = X' 

function MLJModelInterface.fit(model::MulticlassPerceptronClassifier,
                     verbosity::Int,
                     X,
                     y)

    n_classes   = length(MLJModelInterface.classes(y[1]))
    Xmatrix = _reformat(X)
    n_features = size(Xmatrix, 1)
    decode  = MLJModelInterface.decoder(y[1]) # Storing a decode for the predict method
    y = Int.(MLJModelInterface.int(y))        # Encoding categorical target as array of integers

    is_sparse = issparse(Xmatrix)
    perceptron = MulticlassPerceptronCore(model.element_type,
                                                    n_classes,
                                                    n_features,
                                                    is_sparse);

    ### Fitting code starts
    fit!(perceptron, Xmatrix, y;
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

function MLJModelInterface.predict(model::MulticlassPerceptronClassifier,
                                   fitresult,
                                   Xnew)

    Xmatrix = _reformat(Xnew)

    result, decode = fitresult
    prediction = predict(result, Xmatrix)
    return decode(prediction)
end

function MLJModelInterface.predict(fitresult,
                                   Xnew)

    Xmatrix = _reformat(Xnew)

    result, decode = fitresult
    prediction = predict(result, Xmatrix)
    return decode(prediction)
end

#= =======================
   METADATA FOR ALL MODELS
   ======================= =#

descr_(M) = descr(M) *
    "\n→ based on [MulticlassPerceptron](https://github.com/davidbp/MulticlassPerceptron.jl)" *
    "\n→ do `@load $(MLJModelInterface.name(M)) pkg=\"MulticlassPerceptron\" to use the model.`" *
    "\n→ do `?$(MLJModelInterface.name(M))` for documentation."

lp_(M) = "MulticlassPerceptron.$(MLJModelInterface.name(M))"

MLJModelInterface.metadata_model(MulticlassPerceptronClassifier,
    input=Union{MLJModelInterface.Table(MLJModelInterface.Continuous),
                AbstractMatrix{MLJModelInterface.Continuous}},
    target=AbstractVector{<:MLJModelInterface.Finite},
    weights=false,
    descr=descr_(MulticlassPerceptronClassifier),
    path=lp_(MulticlassPerceptronClassifier))
