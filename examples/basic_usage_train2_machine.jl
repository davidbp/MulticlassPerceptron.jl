
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

## MLJBase.fit needs as input X array
#X = copy(MLJ.matrix(X)')

## Train the model
println("\nStart Learning\n")
#fitresult, _ , _  = MLJBase.fit(perceptron, 1, X, y)
fit!(perceptron_machine)

println("\nLearning Finished\n")

## Make predictions
y_hat_train = MLJBase.predict(perceptron_machine, X)
# ERROR: MethodError: no method matching adjoint(::DataFrame)


## Evaluate the model
println("Results:")
println("Train accuracy:", mean(y_hat_train .== y))
println("\n")
