### Code suggestions

 * `num_features_and_observations` can  just be `reverse(size(X))` and the other one `size(X)`, also given  how you use it, it's probably best to just use  `size` itself but YOLO . 
 * DONE`
* you  could make your `MulticlassPerceptronClassifier` object use the elemnt type as a  parametric type and not carry around the element type

```julia
    if Xnew isa AbstractArray
        Xnew  = MLJBase.matrix(Xnew)
    elseif Tables.istable(Xnew) 
        Xnew  = MLJBase.matrix(Xnew, transpose=true)
    end
```

replace with

```julia
if Tables.istable(Xnew)
  Xnew = ...
end
```

* ​	DONE			

* for  readability it'd be better to put `fit`  before `predict`
  * DONE
* same comment with the unnecessary  abstractarray test in `fit`
  * DONE
* please do  not explicitly  specify  that  `y::CategoricalArray` (this  is done upstream)
  * DONE
* please do not print lines that are not constrained by the verbosity element. and definitely  for non  informative lines like `"training"`
  * DONE

Why are there two  implementations of `predict` ? DONE

### Tests

You currently  don't have tests, I would  suggest removing all notebooks and converting them in tests but even if you keep  notebooks, please add rigorous tests, we  will only  add  tested packages to the  registry. 

In the  tests, please test the interface (again have a look at MLJLinearModels for examples)

In  the tests please also consider benchmarking against sklearn's MLP implementation

### Next step

Once all that is done, the steps are:

* release your package to the Julia registry
* open an  issue on MLJModels for us to  add your registered package to  the registry