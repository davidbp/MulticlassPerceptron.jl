
#using MulticlassPerceptron
using Statistics
using MLJ, MLJBase, CategoricalArrays

# We use flux only to get the MNIST
using Flux, Flux.Data.MNIST

push!(LOAD_PATH, "./src/")
using MulticlassPerceptron

train_imgs = MNIST.images(:train)   # size(train_imgs) -> (60000,)
test_imgs  = MNIST.images(:test)    # size(test_imgs) -> (10000,)
train_x    = Float32.(hcat(reshape.(train_imgs, :)...)) # size(train_x) -> (784, 60000)
test_x     = Float32.(hcat(reshape.(test_imgs, :)...)) # size(test_x)   -> (784, 60000)
println("\nMNIST Dataset Example")

## Prepare data
train_y = MNIST.labels(:train) .+ 1;
test_y  = MNIST.labels(:test)  .+ 1;

## Encode targets as CategoricalArray objects
train_y = CategoricalArray(train_y)
test_y  = CategoricalArray(test_y)

## Define model and train it
n_features = size(train_x, 1);
n_classes  = length(unique(train_y));
perceptron =  MulticlassPerceptron.MulticlassPerceptronClassifier(n_epochs=50; f_average_weights=true)

## Train the model
println("\nStart Learning\n")
fitresult, _ , _  = MLJBase.fit(perceptron, 1, train_x, train_y) # If train_y is a CategoricalArray

println("\nLearning Finished\n")

## Make predictions
y_hat_train = MLJBase.predict(fitresult, train_x)
y_hat_test  = MLJBase.predict(fitresult, test_x);

## Evaluate the model
println("Results:")
println("Train accuracy:", mean(y_hat_train .== train_y))
println("Test accuracy:",  mean(y_hat_test  .== test_y))
println("\n")
