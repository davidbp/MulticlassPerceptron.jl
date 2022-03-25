
using Statistics

# We use flux only to get the MNIST
using Flux, Flux.Data.MNIST
#using CategoricalArrays

# Load MulticlassPerceptron
#push!(LOAD_PATH, "../src/") 
using MulticlassPerceptron
<<<<<<< HEAD:examples/core_usage_train.jl
=======

println("\nMNIST Dataset, MulticlassPerceptronCore")

>>>>>>> MLModelinterface:test/sklearn_comparisson/mnist_julia.jl

function load_MNIST( ;array_eltype::DataType=Float32, verbose::Bool=true)

    if verbose
        time_init = time()
        println("MNIST Dataset Loading...")
    end
    train_imgs = MNIST.images(:train)                             # size(train_imgs) -> (60000,)
    test_imgs  = MNIST.images(:test)                              # size(test_imgs)  -> (10000,)
    train_x    = array_eltype.(hcat(reshape.(train_imgs, :)...))  # size(train_x)    -> (784, 60000)
    test_x     = array_eltype.(hcat(reshape.(test_imgs, :)...))   # size(test_x)     -> (784, 60000)

    ## Prepare data
    train_y = MNIST.labels(:train) .+ 1;
    test_y  = MNIST.labels(:test)  .+ 1;

<<<<<<< HEAD:examples/core_usage_train.jl
=======
    ## CategoricalArray are not needed for the MulticlassPerceptronCore 
    #train_y = CategoricalArray(train_y)
    #test_y  = CategoricalArray(test_y)

>>>>>>> MLModelinterface:test/sklearn_comparisson/mnist_julia.jl
    if verbose
        time_taken = round(time()-time_init; digits=3)
        println("MNIST Dataset Loaded, it took $time_taken seconds")
    end
    return train_x, train_y, test_x, test_y
end

println("\nLoading data")
train_x, train_y, test_x, test_y = load_MNIST( ;array_eltype=Float32, verbose=true)

<<<<<<< HEAD:examples/core_usage_train.jl
## Define model and train it
n_features = size(train_x, 1);
n_classes  = length(unique(train_y));
=======
n_features = size(train_x, 1);
n_classes  = length(unique(train_y));


## Define model and train it
>>>>>>> MLModelinterface:test/sklearn_comparisson/mnist_julia.jl
perceptron = MulticlassPerceptronCore(Float32,
                                      n_classes,
                                      n_features,
                                      false)
<<<<<<< HEAD:examples/core_usage_train.jl
=======

println("\nTypes and shapes before calling fit!(perceptron, train_x, train_y)")
@show typeof(perceptron)
@show typeof(train_x)
@show typeof(train_y)
@show size(train_x)
@show size(train_y)
@show size(test_x)
@show size(test_y)
@show n_features
@show n_classes
>>>>>>> MLModelinterface:test/sklearn_comparisson/mnist_julia.jl

## Train the model
println("\n\nStart Learning")
time_init = time()
<<<<<<< HEAD:examples/core_usage_train.jl
fit!(perceptron, train_x, train_y, f_average_weights=true, n_epochs=50) 
=======
fit!(perceptron, train_x, train_y; n_epochs=100) 
>>>>>>> MLModelinterface:test/sklearn_comparisson/mnist_julia.jl
time_taken = round(time()-time_init; digits=3)
println("Learning took $time_taken seconds")

## Make predictions
y_hat_train = predict(perceptron, train_x)
y_hat_test  = predict(perceptron, test_x);

## Evaluate the model
println("\nResults:")
println("Train accuracy:", round(mean(y_hat_train .== train_y), digits=3) )
println("Test accuracy:",  round(mean(y_hat_test  .== test_y), digits=3) ) 
println("\n")
