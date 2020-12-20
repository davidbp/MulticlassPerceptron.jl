
using Statistics

# We use flux only to get the MNIST
using Flux, Flux.Data.MNIST
using CategoricalArrays

# Load MulticlassPerceptron
using MulticlassPerceptron

function load_MNIST( ;array_eltype::DataType=Float32, verbose::Bool=true)

    if verbose
        time_init = time()
        println("\nMNIST Dataset Loading...")
    end
    train_imgs = MNIST.images(:train)                             # size(train_imgs) -> (60000,)
    test_imgs  = MNIST.images(:test)                              # size(test_imgs)  -> (10000,)
    train_x    = array_eltype.(hcat(reshape.(train_imgs, :)...))  # size(train_x)    -> (784, 60000)
    test_x     = array_eltype.(hcat(reshape.(test_imgs, :)...))   # size(test_x)     -> (784, 60000)

    ## Prepare data
    train_y = MNIST.labels(:train) .+ 1;
    test_y  = MNIST.labels(:test)  .+ 1;

    if verbose
        time_taken = round(time()-time_init; digits=3)
        println("\nMNIST Dataset Loaded, it took $time_taken seconds")
    end
    return train_x, train_y, test_x, test_y
end

println("\nLoading data\n")
train_x, train_y, test_x, test_y = load_MNIST( ;array_eltype=Float32, verbose=true)

## Define model and train it
n_features = size(train_x, 1);
n_classes  = length(unique(train_y));
perceptron = MulticlassPerceptronCore(Float32,
                                      n_classes,
                                      n_features,
                                      false)

## Train the model
println("\nStart Learning\n")
time_init = time()
fit!(perceptron, train_x, train_y, f_average_weights=true, n_epochs=50) 
time_taken = round(time()-time_init; digits=3)

println("\nLearning took $time_taken seconds\n")

## Make predictions
y_hat_train = predict(perceptron, train_x)
y_hat_test  = predict(perceptron, test_x);

## Evaluate the model
println("Results:")
println("Train accuracy:", mean(y_hat_train .== train_y))
println("Test accuracy:",  mean(y_hat_test  .== test_y))
println("\n")
