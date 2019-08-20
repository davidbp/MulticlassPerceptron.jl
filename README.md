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

Executing `basic_usage_train.jl` you should get

```
Start Learning

Epoch: 50 	 Accuracy: 0.898
Learning Finished

Results:
Train accuracy:0.9359333333333333
Test accuracy:0.927
```

If this works then you can already use `MulticlassPerceptron` models!



### Basic usage 

This code snippet loads the MNIST Dataset and saves the classes as a `CategoricalArray`

```julia
using MLDatasets
using CategoricalArrays

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

