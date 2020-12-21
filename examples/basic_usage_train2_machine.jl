
#using MulticlassPerceptron
using Statistics
using MLJ, MLJBase, CategoricalArrays
using Random

#push!(LOAD_PATH, "../src/") ## Uncomment if MulticlassPerceptron not installed
using MulticlassPerceptron

## Prepare data
using RDatasets                                      # this is unreasonably slow
iris = dataset("datasets", "iris"); # a DataFrame    # this is unreasonably slow
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

## Define a Machine
perceptron_machine = machine(perceptron, X, y)

println("\nTypes and shapes before calling fit!(perceptron_machine)")
@show typeof(perceptron_machine)
@show typeof(X)
@show typeof(y)
@show size(X)
@show size(y)
@show n_features
@show n_classes

## Train the model
println("\nStart Learning\n")
time_init = time()
#fitresult, _ , _  = MLJBase.fit(perceptron, 1, X, y)
fit!(perceptron_machine)
time_taken = round(time()-time_init; digits=3)
println("Learning took $time_taken seconds\n")

## Make predictions
y_hat_train = predict(perceptron_machine, X)

## Evaluate the model
println("\nResults:")
println("Train accuracy:", round(mean(y_hat_train .== y), digits=3) )
println("\n")