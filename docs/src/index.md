
# Getting started


The basic struct storing the weights of a Multiclass Perceptorn is:
```@docs
MulticlassPerceptronClassifierCore{T}
```

This function has a constructor
```@docs
MulticlassPerceptronClassifierCore(T::Type, n_classes::Int, n_features::Int,  is_sparse::Bool)
```


The function used to train a MulticlassPerceptronCore struct is:
```@docs
fit!
```

# Interface with MLJ

The struct `MulticlassPerceptronClassifier` can be created instanciated in order to make a model
suitable to work with MLJ.
