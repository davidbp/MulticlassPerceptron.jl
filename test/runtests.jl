using Test
using Random
using MLJBase
using CategoricalArrays

#push!(LOAD_PATH, "../src/")
using MulticlassPerceptron

@testset "MulticlassPerceptronCore" begin

    @testset "n=100, p=2, n_classes=2, sparse=false, f_average_weights=true" begin
        Random.seed!(6161)
        n, p, n_classes, sparse, f_average_weights = 100, 2, 2, false, true
        X = randn(p, n)
        y = rand(1:n_classes, n)

        perceptron_core = MulticlassPerceptronCore(Float64, n_classes, p, sparse)
        fit!(perceptron_core, X, y; n_epochs=10, f_average_weights=true)
        ŷ = MulticlassPerceptron.predict(perceptron_core, X)
        @test length(ŷ)==length(y)
    end

    @testset "n=100, p=10, n_classes=3, sparse=false, f_average_weights=true" begin
        Random.seed!(6161)
        n, p, n_classes, sparse, f_average_weights = 100, 10, 3, false, true
        X = randn(p, n)
        y = rand(1:n_classes, n)

        perceptron_core = MulticlassPerceptronCore(Float64, n_classes, p, sparse)
        fit!(perceptron_core, X, y; n_epochs=10, f_average_weights=true)
        ŷ = MulticlassPerceptron.predict(perceptron_core, X)
        @test length(ŷ)==length(y)
    end

end

@testset "MulticlassPerceptronClassifier" begin

    @testset "n=100, p=2, n_classes=2" begin
        Random.seed!(6161)
        n, p, n_classes, verbosity = 100, 2, 2, 0
        X = randn(n, p)
        y = CategoricalArray(rand(1:n_classes, n))

        X = MLJBase.table(X)
        perceptron = MulticlassPerceptronClassifier(n_epochs=10;
                                                f_average_weights=true)
        fitresult, = fit(perceptron, verbosity, X, y)
        ŷ          = MLJBase.predict(fitresult, X)
        @test length(ŷ)==length(y)
    end

    @testset "n=100, p=10, n_classes=3" begin
        Random.seed!(6161)
        n, p, n_classes, verbosity = 100, 10, 3, 0
        X = randn(n, p)
        y = CategoricalArray(rand(1:n_classes, n))

        X = MLJBase.table(X)
        perceptron = MulticlassPerceptronClassifier(n_epochs=10;
                                                f_average_weights=true)
        fitresult, = fit(perceptron, verbosity, X, y)
        ŷ          = MLJBase.predict(fitresult, X)
        @test length(ŷ)==length(y)
    end

end

using MLJ

@testset "MulticlassPerceptronClassifier machine" begin

    @testset "n=100, p=2, n_classes=2" begin
        Random.seed!(6161)
        n, p, n_classes, verbosity = 100, 2, 2, 0
        X = MLJBase.table(randn(n, p))
        y = CategoricalArray(rand(1:n_classes, n))

        perceptron = MulticlassPerceptronClassifier(n_epochs=10;
                                                    f_average_weights=true)
        perceptron_machine = machine(perceptron, X, y)
        fit!(perceptron_machine)
        ŷ = MLJBase.predict(perceptron_machine, X)
        @test length(ŷ)==length(y)
    end

    @testset "n=100, p=10, n_classes=3" begin
        Random.seed!(6161)
        n, p, n_classes, verbosity = 100, 10, 3, 0
        X = MLJBase.table(randn(n, p))
        y = CategoricalArray(rand(1:n_classes, n))

        perceptron = MulticlassPerceptronClassifier(n_epochs=10;
                                                    f_average_weights=true)
        perceptron_machine = machine(perceptron, X, y)
        fit!(perceptron_machine)
        ŷ = MLJBase.predict(perceptron_machine, X)
        @test length(ŷ)==length(y)
    end

end
