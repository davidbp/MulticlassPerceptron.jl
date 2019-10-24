module MulticlassPerceptron


# using MetadataTools,  DocStringExtensions
using Random: shuffle, MersenneTwister
using DataFrames

# export MulticlassPerceptronClassifier, fit!, predict
using LinearAlgebra: mul!
#using SparseArrays

# Add explicit case for table data not to use sparse format
import SparseArrays.issparse
issparse(X::DataFrame) = false

import MLJBase
using MLJ
using CategoricalArrays
using SparseArrays

export MulticlassPerceptronClassifierCore, predict, fit

num_features_and_observations(X::DataFrame)     = (size(X,2), size(X,1))
num_features_and_observations(X::AbstractArray) = (size(X,1), size(X,2))

############################################################################
## Defining MulticlassPerceptronClassifier
############################################################################

mutable struct MulticlassPerceptronClassifier <: MLJBase.Deterministic
    n_epochs::Int
    n_epoch_patience::Int
    f_average_weights::Bool
    f_shuffle_data::Bool
    element_type::DataType
end


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
    if Xnew isa AbstractDataFrame
        xnew  = MLJBase.matrix(Xnew)
        xnew  = copy(Xnew')           # In this case I transpose because I assume a user passed examples as rows
    else
        xnew  = MLJBase.matrix(Xnew)
    end

    result, decode = fitresult
    prediction = predict(result, xnew)
    return decode(prediction)
end



function MLJBase.fit(model::MulticlassPerceptronClassifier,
                     verbosity::Int,
                     X,
                     y)

    @assert y isa CategoricalArray "typeof(y)=$(typeof(y)) but typeof(y) should be a CategoricalArray"

    n_classes      = length(unique(y))
    classes_seen   = unique(y)
    n_features, _  = num_features_and_observations(X)

    # Function fit!(perceptron, X, y) expects size(X) = n_features x n_observations
    if X isa AbstractDataFrame
        X  = MLJBase.matrix(X)
        X  = copy(X')           # In this case I transpose because I assume a user passed examples as rows
    else X isa AbstractArray
        X  = MLJBase.matrix(X)  # (TODO: rethink this, MLJ will assume allways examples as rows) 
                                # In this case I assume examples are the columns of X becuase the fit! with MulticlassPerceptronClassifierCore
                                # precisely iterates over examples as columns
    end

    decode  = MLJBase.decoder(y[1]) # for the predict method
    y = Int.(MLJ.int(y))            # Encoding categorical target as array of integers

    is_sparse = issparse(X)
    perceptron = MulticlassPerceptronClassifierCore( model.element_type, n_classes,
                                                          n_features, is_sparse);
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

    #> return package-specific statistics (eg, feature rankings,
    #> internal estimates of generalization error) in `report`, which
    #> should be a named tuple with the same type every call (can have
    #> empty values)
    return fitresult, cache, report
end



############################################################################
## Defining MulticlassPerceptronClassifierCore
############################################################################

mutable struct MulticlassPerceptronClassifierCore{T}
    W::AbstractMatrix{T}
    b::AbstractVector{T}
    n_classes::Int
    n_features::Int
    is_sparse::Bool
end


"""
Function to build an empty **`MulticlassPerceptronClassifierCore`**.

The option **`is_parse`**  stores the weights of the model as sparse if `isparse=True`, as standard Array otherwise.

"""
function MulticlassPerceptronClassifierCore(T::Type, n_classes::Int, n_features::Int, is_sparse::Bool)

    if is_sparse==false
        return MulticlassPerceptronClassifierCore{T}(rand(T, n_features, n_classes),
                                                                                       zeros(T, n_classes),
                                                                                       n_classes,
                                                                                       n_features,
                                                                                       is_sparse)
    else
        return  MulticlassPerceptronClassifierCore{T}(sparse(rand(T, n_features, n_classes)),
                                                            spzeros(T, n_classes),
                                                            n_classes,
                                                            n_features,
                                                            is_sparse)
    end
end

## Auxiliar methods

"""
Predicts the class for a given input in a `MulticlassPerceptronClassifier`.
The placeholder is used to avoid allocating memory for each matrix-vector multiplication.

- Returns the predicted class.
"""
function predict_with_placeholder(h::MulticlassPerceptronClassifierCore, x::AbstractVector, class_placeholder::AbstractVector)
    #@fastmath class_placeholder .= At_mul_B!(class_placeholder, h.W, x) .+ h.b
    class_placeholder .= mul!(class_placeholder, transpose(h.W), x)  .+ h.b
    return argmax(class_placeholder)
end


"""
Function to compute the accuracy between `y` and `y_hat`.
"""
function accuracy(y::AbstractVector, y_hat::AbstractVector)
    acc = 0.
    @fastmath for k = 1:length(y)
            @inbounds  acc += y[k] == y_hat[k]
    end
    return acc/length(y_hat)
end

function predict(h::MulticlassPerceptronClassifierCore, x::AbstractVector, class_placeholder::AbstractVector)
    class_placeholder .= mul!(class_placeholder, transpose(h.W), x)  .+ h.b
    return argmax(class_placeholder)
end

"""
Function to predict the class for a given input batch.
- Returns the predicted class.
"""
function predict(h::MulticlassPerceptronClassifierCore, X::AbstractMatrix)
    predictions = zeros(Int64, size(X, 2))
    class_placeholder = zeros(eltype(h.W), h.n_classes)

    @inbounds for m in 1:length(predictions)
        predictions[m] = predict(h, view(X,:,m), class_placeholder)
    end

    return predictions
end

function MLJBase.predict(fitresult::Tuple{MulticlassPerceptronClassifierCore, MLJBase.CategoricalDecoder}, Xnew)

    # Function fit!(perceptron, X, y) expects size(X) = n_features x n_observations
    if Xnew isa AbstractDataFrame
        Xnew  = MLJBase.matrix(Xnew)
        xnew  = copy(Xnew')           # In this case I transpose because I assume a user passed examples as rows
    else 
        xnew  = MLJBase.matrix(Xnew)
    end

    #xnew = MLJBase.matrix(Xnew)
    result, decode = fitresult
    prediction = predict(result, xnew)
    return decode(prediction)
end


"""
>    fit!(h::MulticlassPerceptronClassifier,
>         X::Array,
>         y::Array;
>         n_epochs=50,
>         learning_rate=0.1,
>         print_flag=false,
>         compute_accuracy=true,
>         seed=Random.seed!(1234),
>         pocket=false,
>         shuffle_data=false)

##### Arguments

- **`h`**, (MulticlassPerceptronClassifier{T} type), Multiclass perceptron.
- **`X`**, (Array{T,2} type), data contained in the columns of X.
- **`y`**, (Vector{T} type), class labels (as integers from 1 to n_classes).

##### Keyword arguments

- **`n_epochs`**, (Integer type), number of passes (epochs) through the data.
- **`learning_rate`**, (Float type), learning rate (The standard perceptron is with learning_rate=1.)
- **`compute_accuracy`**, (Bool type), if `true` the accuracy is computed at the end of every epoch.
- **`print_flag`**, (Bool type), if `true` the accuracy is printed at the end of every epoch.
- **`seed`**, (MersenneTwister type), seed for the permutation of the datapoints in case there the data is shuffled.
- **`pocket`** , (Bool type), if `true` the best weights are saved (in the pocket) during learning.
- **`shuffle_data`**, (Bool type),  if `true` the data is shuffled at every epoch (in reality we only shuffle indicies for performance).
"""
function fit!(h::MulticlassPerceptronClassifierCore, X::AbstractArray, y::AbstractVector;
              verbosity=0,
              n_epochs=50,
              learning_rate=1.,
              f_average_weights=false,
              compute_accuracy=true,
              seed=MersenneTwister(1234),
              f_shuffle_data=false)

    ## n_features, n_observations = size(X)
    n_features, n_observations = num_features_and_observations(X)
    
    @assert n_observations == length(y) "n_observations = $n_observations but length(y)=$(length(y))"

    scores = []
    T = eltype(X)
    counter           = 0
    learning_rate     = T(learning_rate)
    class_placeholder = zeros(T, h.n_classes)
    y_preds           = zeros(Int32, n_observations)

    data_indices      = Array(1:n_observations)
    max_acc           = zero(T)

    if f_average_weights
        W_average =  zeros(T, h.n_features, h.n_classes)
        b_average =  zeros(T, h.n_classes)
    end

    @fastmath for epoch in 1:n_epochs

        n_mistakes = 0
        if f_shuffle_data
            shuffle!(seed, data_indices)
        end

        @inbounds for m in data_indices
            x     = view(X, :, m);
            y_hat = predict_with_placeholder(h, x, class_placeholder)

            if y[m] != y_hat
                n_mistakes += 1
                ####  wij ← wij − η (yj −tj) · xi
                h.W[:, y[m]]  .= h.W[:, y[m]]  .+ learning_rate .* x
                h.b[y[m]]      = h.b[y[m]]      + learning_rate
                h.W[:, y_hat] .= h.W[:, y_hat] .- learning_rate .* x
                h.b[y_hat]     = h.b[y_hat]     - learning_rate

                if f_average_weights == true
                    counter_learning_rate = counter * learning_rate
                    W_average[:, y[m]]   .= W_average[:, y[m]]  .+ counter_learning_rate .* x
                    b_average[y[m]]       = b_average[y[m]]      + counter_learning_rate
                    W_average[:, y_hat]  .= W_average[:, y_hat] .- counter_learning_rate .* x
                    b_average[y_hat]      = b_average[y_hat]     - counter_learning_rate
                end
            end
            counter +=1
        end
            
        acc = (n_observations - n_mistakes)/n_observations
        # push!(scores, acc) maybe it would be nice to return an array with monitoring metrics to
        # allow users to decide if the model has converged

        if verbosity ==1
            print("\r\u1b[K")
            print("Epoch: $(epoch) \t Accuracy: $(round(acc; digits=3))")
        elseif verbosity ==2
            println("Epoch: $(epoch) \t Accuracy: $(round(acc; digits=3))")
        end
    end

    if f_average_weights == true
        h.W .= h.W  .- W_average./counter
        h.b .= h.b  .- b_average./counter
    end

end

end # module
