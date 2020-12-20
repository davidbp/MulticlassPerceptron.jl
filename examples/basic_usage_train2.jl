
#using MulticlassPerceptron
using Statistics
using MLJBase, CategoricalArrays
using Random

#push!(LOAD_PATH, "../src/") ## Uncomment if MulticlassPerceptron not installed
using MulticlassPerceptron

## Prepare data
using RDatasets                                
iris = dataset("datasets", "iris"); 
#using RCall
#iris = R"iris" |> rcopy

scrambled = shuffle(1:size(iris, 1))
X = iris[scrambled, 1:4];
y = iris[scrambled, 5];
println("\nIris Dataset Example")

## Encode targets as CategoricalArray objects
y = CategoricalArray(y)

## Define model and train it
n_features = size(X, 2);
n_classes  = length(unique(y));
perceptron = MulticlassPerceptronClassifier(n_epochs=50; f_average_weights=true)

## MLJBase.fit needs as input X array
X = copy(matrix(X)')

## Train the model
println("\nStart Learning\n")
fitresult, _  = fit(perceptron, 1, X, y)

println("\nLearning Finished\n")

## Make predictions
y_hat_train = MLJBase.predict(fitresult, X)
# MLJBase.predict(fitresult[1], MLJBase.matrix(X)') # this works

## Evaluate the model
println("Results:")
println("Train accuracy:", mean(y_hat_train .== y))
println("\n")
