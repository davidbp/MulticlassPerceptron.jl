
using Statistics
using Random

# Load MulticlassPerceptron
#push!(LOAD_PATH, "../src/") 
using MulticlassPerceptron

println("\nIris Dataset, MulticlassPerceptronCore")

## Prepare data
using RDatasets                                
iris = dataset("datasets", "iris"); 

scrambled = shuffle(1:size(iris, 1))
X = iris[scrambled, 1:4];
y = iris[scrambled, 5];
X = Array(X)'

## Encode targets as CategoricalArray objects
y = CategoricalArray(y)
y = Array{Int64}(y.refs)

n_features = size(X, 1);
n_classes  = length(unique(y));


## Define model and train it
perceptron = MulticlassPerceptronCore(Float32,
                                      n_classes,
                                      n_features,
                                      false)

println("\nTypes and shapes before calling fit!(perceptron, train_x, train_y)")
@show typeof(perceptron)
@show typeof(X)
@show typeof(y)
@show size(X)
@show size(y)
@show n_features
@show n_classes

## Train the model
println("\n\nStart Learning")
time_init = time()
fit!(perceptron, X, y) 
time_taken = round(time()-time_init; digits=3)
println("Learning took $time_taken seconds")

## Make predictions
y_hat_train = predict(perceptron, X)

## Evaluate the model
println("\nResults:")
println("Train accuracy:", round(mean(y_hat_train .== y), digits=3) )
println("\n")
