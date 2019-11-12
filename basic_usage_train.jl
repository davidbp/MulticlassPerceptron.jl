
#using MulticlassPerceptron
using Statistics
using MLJ, MLJBase, CategoricalArrays

# We use flux only to get the MNIST
using Flux, Flux.Data.MNIST

# Load MulticlassPerceptron
push!(LOAD_PATH, "./src/")
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
    
    ## Encode targets as CategoricalArray objects
    train_y = CategoricalArray(train_y)
    test_y  = CategoricalArray(test_y)
    
    if verbose
        time_taken = round(time()-time_init; digits=3)
        println("\nMNIST Dataset Loaded, it took $time_taken seconds")
    end
    return train_x, train_y, test_x, test_y
end

println("\nLoading data\n")
train_x, train_y, test_x, test_y =  load_MNIST( ;array_eltype=Float32, verbose=true)


## Define model and train it
n_features = size(train_x, 1);
n_classes  = length(unique(train_y));
perceptron = MulticlassPerceptron.MulticlassPerceptronClassifier(n_epochs=50; f_average_weights=true)


## Train the model
println("\nStart Learning\n")
time_init = time()
fitresult, _ , _  = MLJBase.fit(perceptron, 1, train_x, train_y) # 
time_taken = round(time()-time_init; digits=3)

println("\nLearning took $time_taken seconds\n")

## Make predictions
y_hat_train = MLJBase.predict(fitresult, train_x)
y_hat_test  = MLJBase.predict(fitresult, test_x);

## Evaluate the model
println("Results:")
println("Train accuracy:", mean(y_hat_train .== train_y))
println("Test accuracy:",  mean(y_hat_test  .== test_y))
println("\n")
