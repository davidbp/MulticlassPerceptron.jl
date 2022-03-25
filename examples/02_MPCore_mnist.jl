
using Statistics

using MLDatasets
using CategoricalArrays

# Load MulticlassPerceptron
#push!(LOAD_PATH, "../src/") 
using MulticlassPerceptron

println("\nMNIST Dataset, MulticlassPerceptronCore")
function load_MNIST( ;array_eltype::DataType=Float32, as_image::Bool=false, verbose::Bool=true)

    if verbose
        time_init = time()
        println("MNIST Dataset Loading...")
    end
    train_x, train_y = MNIST.traindata()
    test_x,  test_y  = MNIST.testdata()
        
    if as_image == false
        train_x = reshape(train_x,(28*28, 60_000))
        test_x = reshape(test_x,(28*28, 10_000))        
    end
    
    ## Prepare data
    train_y = Int.(train_y .+ 1);
    test_y  = Int.(test_y .+ 1);

    #train_y = CategoricalArray(train_y)
    #test_y  = CategoricalArray(test_y)
    
    
    if verbose
        time_taken = round(time()-time_init; digits=3)
        println("MNIST Dataset Loaded, it took $time_taken seconds")
    end
    return array_eltype.(train_x),train_y,  array_eltype.(test_x), test_y
end

println("\nLoading data")
train_x, train_y, test_x, test_y = load_MNIST( ;array_eltype=Float32, verbose=true)

n_features = size(train_x, 1);
n_classes  = length(unique(train_y));


## Define model and train it
perceptron = MulticlassPerceptronCore(Float32,
                                      n_classes,
                                      n_features,
                                      false)

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

## Train the model
println("\n\nStart Learning")
time_init = time()
fit!(perceptron, train_x, train_y; n_epochs=100) 
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
