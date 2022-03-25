#using MulticlassPerceptron
using Statistics
using MLJ, MLJBase, CategoricalArrays

# We use flux only to get the MNIST
using Flux, Flux.Data.MNIST

# Load MulticlassPerceptron
#push!(LOAD_PATH, "../src/") ## Uncomment if MulticlassPerceptron not installed
using MulticlassPerceptron

println("\nMNIST Dataset, Machine with a MulticlassPerceptronClassifier")


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

train_x, train_y, test_x, test_y = load_MNIST( ;array_eltype=Float32, verbose=true)

## Define model and train it
n_features = size(train_x, 1);
n_classes  = length(unique(train_y));
perceptron = MulticlassPerceptronClassifier(n_epochs=50; f_average_weights=true)

## Define a Machine
#train_x = MLJBase.table(train_x')  # machines can work with Tables.Table or DataFrame objects              
#test_x = MLJBase.table(test_x')   # machines can work with Tables.Table or DataFrame objects              
train_x = train_x'                  # machines expect data to be in rows
test_x = test_x'                    # machines expect data to be in rows

perceptron_machine = machine(perceptron, train_x, train_y)   

println("\nTypes and shapes before calling fit!(perceptron_machine)")
@show typeof(perceptron_machine)
@show typeof(train_x)
@show typeof(train_y)
@show size(train_x)
@show size(train_y)
@show size(test_x)
@show size(test_y)
@show n_features
@show n_classes

## Train the model
println("\nStart Learning\n")
time_init = time()
#fitresult, _ , _  = MLJBase.fit(perceptron, 1, train_x, train_y) # If train_y is a CategoricalArray
fit!(perceptron_machine)
time_taken = round(time()-time_init; digits=3)
println("\nLearning took $time_taken seconds\n")

## Make predictions
y_hat_train = MLJBase.predict(perceptron_machine, train_x)
y_hat_test  = MLJBase.predict(perceptron_machine, test_x);

## Evaluate the model
println("Results:")
println("Train accuracy:", round(mean(y_hat_train .== train_y), digits=3) )
println("Test accuracy:",  round(mean(y_hat_test  .== test_y), digits=3) ) 
println("\n")

