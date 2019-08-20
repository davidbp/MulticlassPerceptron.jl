
using MLDatasets, Statistics
#using MulticlassPerceptron
using CategoricalArrays
using MLJ
using MLJBase

push!(LOAD_PATH, "./src/")
using MulticlassPerceptron

## Prepare data
train_x, train_y = MLDatasets.MNIST.traindata();
test_x, test_y   = MLDatasets.MNIST.testdata();
train_x = Float32.(train_x);
test_x  = Float32.(test_x);
train_y = train_y .+ 1;
test_y  = test_y  .+ 1;
train_x = reshape(train_x, 784, 60000);
test_x  = reshape(test_x,  784, 10000);

## Encode targets as CategoricalArray objects
train_y = CategoricalArray(train_y)
test_y  = CategoricalArray(test_y)

## Define model and train it
scores = []
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
