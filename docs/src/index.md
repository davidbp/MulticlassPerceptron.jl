


# Multiclass Perceptron


The basic constructor needed to create a `MulticlassPerceptronCore` struct is:
```julia
MulticlassPerceptronClassifierCore(T::Type,
                                   n_classes::Int,
                                   n_features::Int,
                                   is_sparse::Bool)
```


This constructor creates a struct that stores the weight matrix as well as the bias vector of the model.

```julia
MulticlassPerceptronClassifierCore{T}
```


The function used to train a MulticlassPerceptronCore struct is `fit!`:
```@docs
fit!
```

# Interface with MLJ

Instead of using `MulticlassPerceptronClassifierCore`, the package provides the struct  `MulticlassPerceptronClassifier`  which is compatible with the  MLJ interface. 


#### Example of `MLJ.machine` containing a MulticlassPerceptronClassifier


First, we can create a `perceptron` object

```julia
perceptron = MulticlassPerceptron.MulticlassPerceptronClassifier(n_epochs=50; 
                                                                 f_average_weights=true)
```

Then we can wrap the model and the data into a machine and fit the machine:

```julia
# Define a Machine 
# A machine expects typeof(3.) <: Union{Tables.Table, DataFrame} (not an AbstractArray)
perceptron_machine = machine(perceptron, train_x, train_y) 

# Train the model
MLJBase.fit!(perceptron_machine)
```

