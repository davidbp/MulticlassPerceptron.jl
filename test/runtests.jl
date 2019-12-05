using Test
using Random
using MLJBase
using CategoricalArrays

#push!(LOAD_PATH, "../src/")
using MulticlassPerceptron


@testset "multiclass_perceptron_MLJBase_integration" begin
    Random.seed!(6161)

    n, p = 100, 2
    n_classes = 2
    X = randn(n, p)
    y = CategoricalArray(rand(1:n_classes, n))

    Xt = MLJBase.table(X)
    println("training MulticlassPerceptron with $n examples $p features and $n_classes classes")
    percep = MulticlassPerceptron.MulticlassPerceptronClassifier(n_epochs=10, ; f_average_weights=true)
    fitresult, = MLJBase.fit(percep, 1, Xt, y)
    ŷ          = MLJBase.predict(fitresult, Xt)

    @test length(ŷ)==length(y)
end
