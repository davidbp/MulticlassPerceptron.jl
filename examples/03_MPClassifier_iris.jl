
#using MulticlassPerceptron
using Statistics
using MLJBase, CategoricalArrays
using Random

#push!(LOAD_PATH, "../src/") ## Uncomment if MulticlassPerceptron not installed
using MulticlassPerceptron

## Prepare data
using RDatasets

println("\nIris Dataset, MulticlassPerceptronClassifier")

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

println("\nTypes and shapes before calling fit(perceptron, 1, train_x, train_y)")
@show typeof(perceptron)
@show typeof(X)
@show typeof(y)
@show size(X)
@show size(y)
@show n_features
@show n_classes

## Train the model
println("\nStart Learning")
time_init = time()
fitresult, _  = fit(perceptron, 1, X, y)
time_taken = round(time()-time_init; digits=3)
println("")
@show typeof(fitresult)
println("\nLearning took $time_taken seconds\n")

## Make predictions
y_hat_train = predict(fitresult, X)

## Evaluate the model
println("\nResults:")
println("Train accuracy:", round(mean(y_hat_train .== y), digits=3) )
println("\n")