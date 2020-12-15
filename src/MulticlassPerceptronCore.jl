
# using MetadataTools,  DocStringExtensions
using Random: shuffle, MersenneTwister
using LinearAlgebra: mul!

using StatsBase
#import StatsBase: fit!, predict
import StatsBase: fit!
import MLJModelInterface.predict

# Export methods to be used for MulticlassPerceptronCore
export MulticlassPerceptronCore, predict, fit!


#= #########################################################################
   Defining MulticlassPerceptronCore:

   The code below is MLJ agnostic with the exeption of MLJBase.predict
   Therefore `MulticlassPerceptronCore` can be used outside MLJ

######################################################################### =#

mutable struct MulticlassPerceptronCore{T}
    W::AbstractMatrix{T}
    b::AbstractVector{T}
    n_classes::Int
    n_features::Int
    is_sparse::Bool
end


"""
> function MulticlassPerceptronCore(T::Type,
>                                   n_classes::Int,
>                                   n_features::Int,
>                                   is_sparse::Bool)


Creates an initial  **`MulticlassPerceptronCore`** with random weights.

The boolean  **`isparse=True`** makes the model use sparse weights.
"""
function MulticlassPerceptronCore(T,
                                  n_classes::Int,
                                  n_features::Int,
                                  is_sparse::Bool)

    if is_sparse==false
        return MulticlassPerceptronCore{T}(rand(T, n_features, n_classes),
                                           zeros(T, n_classes),
                                           n_classes,
                                           n_features,
                                           is_sparse)
    else
        return  MulticlassPerceptronCore{T}(sparse(rand(T, n_features, n_classes)),
                                            spzeros(T, n_classes),
                                            n_classes,
                                            n_features,
                                            is_sparse)
    end
end


"""
Predicts the class for a given input using a `MulticlassPerceptronCore`.

The placeholder `class_placeholder` is an array used to avoid allocating memory for each
matrix-vector multiplication. This function is meant to be used while training.

- Returns the predicted class.
"""
function predict_with_placeholder(h::MulticlassPerceptronCore,
                                  x::AbstractVector,
                                  class_placeholder::AbstractVector)

    #@fastmath class_placeholder .= At_mul_B!(class_placeholder, h.W, x) .+ h.b
    class_placeholder .= mul!(class_placeholder, transpose(h.W), x)  .+ h.b
    return argmax(class_placeholder)
end


"""

    accuracy(y::AbstractVector, y_hat::AbstractVector)

Computes the accuracy between `y` and `y_hat`.

# Examples
```julia-repl
julia> accuracy([1, 1, 1], [1, 2, 3])
0.3333333333333333
```
"""
function accuracy(y::AbstractVector, y_hat::AbstractVector)

    acc = 0.
    @fastmath for k = 1:length(y)
            @inbounds  acc += y[k] == y_hat[k]
    end
    return acc/length(y_hat)
end


"""
Function to predict the class for a given input example **`x`** (with placeholder).
- Returns the predicted class.
"""
function predict(h::MulticlassPerceptronCore,
                 x::AbstractVector,
                 class_placeholder::AbstractVector)

    class_placeholder .= mul!(class_placeholder, transpose(h.W), x)  .+ h.b
    return argmax(class_placeholder)
end

"""
Function to predict the class for a given input batch X.
- Returns the predicted class for each element in the X.
"""
function predict(h::MulticlassPerceptronCore, X::AbstractMatrix)
    predictions       = zeros(Int64, size(X, 2))
    class_placeholder = zeros(eltype(h.W), h.n_classes)

    @inbounds for m in 1:length(predictions)
        predictions[m] = predict(h, view(X,:,m), class_placeholder)
    end

    return predictions
end



### TODO MAYBE: Add option pocket as keyword argument
###- **`pocket`** , (Bool type), if `true` the best weights are saved (in the pocket) during learning.


"""
>     fit!(h::MulticlassPerceptronCore,
>          X::AbstractArray,
>          y::AbstractVector;
>          verbosity=0,
>          n_epochs=50,
>          learning_rate=1.,
>          f_average_weights=false,
>          compute_accuracy=true,
>          seed=MersenneTwister(1234),
>          f_shuffle_data=false)

Function to train a MulticlassPerceptronCore model.

##### Arguments

- **`h`**, (MulticlassPerceptronCore{T} type), Multiclass perceptron.
- **`X`**, (Array{T,2} type), data contained in the columns of X.
- **`y`**, (Vector{T} type), class labels (as integers from 1 to n_classes).

##### Keyword arguments

- **`verbosity`**, (Integer type), if `verbosity>0` information is printed.
- **`n_epochs`**, (Integer type), number of passes (epochs) through the data.
- **`learning_rate`**, (Float type), learning rate (The standard perceptron is with learning_rate=1.)
- **`compute_accuracy`**, (Bool type), if `true` the accuracy is computed at the end of every epoch.
- **`seed`**, (MersenneTwister type), seed for the permutation of the datapoints in case there the data is shuffled.
- **`f_shuffle_data`**, (Bool type),  if `true` the data is shuffled at every epoch (in reality we only shuffle indicies for performance).

"""
function fit!(h::MulticlassPerceptronCore,
              X::AbstractArray,
              y::AbstractVector;
              verbosity=0,
              n_epochs=50,
              learning_rate=1.,
              f_average_weights=false,
              compute_accuracy=true,
              seed=MersenneTwister(1234),
              f_shuffle_data=false)


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
