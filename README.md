# Perceptron

Package for training Multiclass Perceptron models.

### Installation

You can clone the package

```julia
using Pkg
Pkg.clone("https://github.com/davidbp/MulticlassPerceptron.jl") 
```

Or use `add` to install the package. Remember to be in `pkg>` mode inside Julia (type `]`).

```
(v1.1) pkg> add "https://github.com/davidbp/MulticlassPerceptron.jl"
```



### Example

In the `examples` folder there are some code examples to test the package.

Executing `julia --project=. ./examples/02_MPCore_mnist.jl` you should get

```
MNIST Dataset, MulticlassPerceptronCore

Loading data
MNIST Dataset Loading...
MNIST Dataset Loaded, it took 1.365 seconds

Types and shapes before calling fit!(perceptron, train_x, train_y)
typeof(perceptron) = MulticlassPerceptronCore{Float32}
typeof(train_x) = Matrix{Float32}
typeof(train_y) = Vector{Int64}
size(train_x) = (784, 60000)
size(train_y) = (60000,)
size(test_x) = (784, 10000)
size(test_y) = (10000,)
n_features = 784
n_classes = 10

Start Learning
Learning took 18.225 seconds

Results:
Train accuracy:0.937
Test accuracy:0.925
```

If this works then you can already use `MulticlassPerceptron` models!

### Core model

The following code shows how to instantiate a struct `MulticlassPerceptronCore` and how to train it.

```julia
using CategoricalArrays
using MulticlassPerceptron
using Statistics

n, p, n_classes, sparse, f_average_weights = 100, 2, 2, false, true
X, y,_ = MulticlassPerceptron.make_blobs(n; centers=n_classes, random_seed=4, return_centers=true)
X = copy(X')

perceptron_core = MulticlassPerceptronCore(Float64, n_classes, p, sparse)
fit!(perceptron_core, X, y; n_epochs=200, f_average_weights=f_average_weights, verbosity=2)
ŷ = MulticlassPerceptron.predict(perceptron_core, X)

println("length(y)=$(length(y))")
println("size(X)=$(size(X))")
println("length(ŷ)=$(length(ŷ))")
println("Accuracy $(mean(ŷ .== y))")
```



### Basic usage 

This code snippet loads the MNIST Dataset and saves the classes as a `CategoricalArray`

```julia

using MulticlassPerceptron
using MLDatasets
using CategoricalArrays

## Load data
train_x, train_y = MNIST.traindata()
test_x,  test_y  = MNIST.testdata()
train_x = reshape(train_x,(28*28, 60_000))
test_x = reshape(test_x,(28*28, 10_000))        
   
## Prepare data
train_y = Int.(train_y .+ 1);
test_y  = Int.(test_y .+ 1);

## Encode targets as CategoricalArray objects
train_y = CategoricalArray(train_y)
test_y  = CategoricalArray(test_y)
```

We can create a `MulticlassPerceptronClassifer` as follows:

```julia
using MulticlassPerceptron
n_features = size(train_x, 1);
n_classes  = length(unique(train_y));
perceptron =  MulticlassPerceptron.MulticlassPerceptronClassifier(n_epochs=50; f_average_weights=true)
```

The function `fit` is used to train the model. The result containing the trained model is kept inside fitresult.

```
fitresult, _ , _  = MLJBase.fit(perceptron, 1, train_x, train_y) 
```

