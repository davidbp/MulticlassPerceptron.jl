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



### Test the code

In the `examples` folder there are some code examples to test the package.

Executing `julia --project=. ./examples/basic_usage_train.jl` you should get

```
 
Loading data


MNIST Dataset Loading...

MNIST Dataset Loaded, it took 0.69 seconds

Start Learning

Epoch: 50 	 Accuracy: 0.899
Learning took 9.635 seconds

Results:
Train accuracy:0.93595
Test accuracy:0.9263

```

If this works then you can already use `MulticlassPerceptron` models!



### Basic usage 

This code snippet loads the MNIST Dataset and saves the classes as a `CategoricalArray`

```julia

using MulticlassPerceptron
using MLDatasets
using CategoricalArrays

## Load data
train_imgs = MNIST.images(:train)   # size(train_imgs) -> (60000,)
test_imgs  = MNIST.images(:test)    # size(test_imgs) -> (10000,)

## Prepare data
train_x    = Float32.(hcat(reshape.(train_imgs, :)...)) # size(train_x) -> (784, 60000)
test_x     = Float32.(hcat(reshape.(test_imgs, :)...)) # size(test_x)   -> (784, 60000)
train_y    = MNIST.labels(:train) .+ 1;
test_y     = MNIST.labels(:test)  .+ 1;

## Encode targets as CategoricalArray objects
train_y = CategoricalArray(train_y)
test_y  = CategoricalArray(test_y)


```

We can create a `MulticlassPerceptronClassifer` as follows :

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

