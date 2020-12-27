using Test
using Random
using CategoricalArrays
using Statistics
import MLJBase

#push!(LOAD_PATH, "../src/")
using MulticlassPerceptron

@testset "MulticlassPerceptronCore (tests model implementation)" begin

    @testset "n=100, p=2, n_classes=2, sparse=false, f_average_weights=true" begin
        Random.seed!(6161)
        n, p, n_classes, sparse, f_average_weights = 100, 2, 2, false, true
        X, y = MulticlassPerceptron.make_blobs(n; centers=n_classes, random_seed=4, return_centers=false)
        X = copy(X')

        perceptron_core = MulticlassPerceptronCore(Float64, n_classes, p, sparse)
        fit!(perceptron_core, X, y; n_epochs=100, f_average_weights=true)
        ŷ = predict(perceptron_core, X)
        @test length(ŷ)==length(y)
        @test mean(ŷ .== y)>0.95
    end

    @testset "n=100, p=10, n_classes=3, sparse=false, f_average_weights=true" begin
        Random.seed!(6161)
        n, p, n_classes, sparse, f_average_weights = 100, 10, 3, false, true
        X = randn(p, n)
        y = rand(1:n_classes, n)

        perceptron_core = MulticlassPerceptronCore(Float64, n_classes, p, sparse)
        fit!(perceptron_core, X, y; n_epochs=10, f_average_weights=true)
        ŷ = predict(perceptron_core, X)
        @test length(ŷ)==length(y)
    end

end

@testset "MulticlassPerceptronClassifier (tests MLJ interface)" begin

    @testset "n=100, p=2, n_classes=2" begin
        Random.seed!(6161)
        n, p, n_classes, verbosity = 100, 2, 2, 0
        X = randn(n, p)
        y = CategoricalArray(rand(1:n_classes, n))

        X = MLJBase.table(X)
        perceptron = MulticlassPerceptronClassifier(n_epochs=10;
                                                f_average_weights=true)
        @test MLJBase.load_path(perceptron) ==
            "MulticlassPerceptron.MulticlassPerceptronClassifier"

        fitresult, = MLJBase.fit(perceptron, verbosity, X, y)
        ŷ          = predict(perceptron, fitresult, X)

        @test length(ŷ)==length(y)
    end

    @testset "n=100, p=10, n_classes=3" begin
        Random.seed!(6161)
        n, p, n_classes, verbosity = 100, 10, 3, 0
        A = randn(n, p)
        y = CategoricalArray(rand(1:n_classes, n))

        X = MLJBase.table(A)
        perceptron = MulticlassPerceptronClassifier(n_epochs=10;
                                                f_average_weights=true)
        Random.seed!(1234)
        fitresult, = MLJBase.fit(perceptron, verbosity, X, y)
        ŷ          = predict(perceptron, fitresult, X)

        @test length(ŷ)==length(y)

        # MLJ user has provided an n x p matrix (raw):
        Random.seed!(1234)
        # should get "warning" that performance is not optimal:
        fitresult, = @test_logs((:info, r"Core multiclass"),
                                MLJBase.fit(perceptron, 1, A, y))
        ŷ2          = predict(perceptron, fitresult, A)
        @test ŷ2 == ŷ

        # MLJ user has provided an n x p matrix (adjoint of a p x n):
        B = permutedims(A)'
        Random.seed!(1234)
        # should not get any extra logging:
        fitresult, = @test_logs MLJBase.fit(perceptron, 1, B, y)
        ŷ3          = predict(perceptron, fitresult, B)
        @test ŷ3 == ŷ

    end

end

@testset "MulticlassPerceptronClassifier machine (tests MLJ machine interface)" begin

    @testset "n=100, p=2, n_classes=2" begin
        Random.seed!(6161)
        n, p, n_classes, verbosity = 100, 2, 2, 0
        X = MLJBase.table(randn(n, p))
        y = CategoricalArray(rand(1:n_classes, n))

        perceptron = MulticlassPerceptronClassifier(n_epochs=10;
                                                    f_average_weights=true)
        perceptron_machine = MLJBase.machine(perceptron, X, y)
        MLJBase.fit!(perceptron_machine)
        ŷ = predict(perceptron_machine, X)
        @test length(ŷ)==length(y)
    end

    @testset "n=100, p=10, n_classes=3" begin
        Random.seed!(6161)
        n, p, n_classes, verbosity = 100, 10, 3, 0
        X = MLJBase.table(randn(n, p))
        y = CategoricalArray(rand(1:n_classes, n))

        perceptron = MulticlassPerceptronClassifier(n_epochs=10;
                                                    f_average_weights=true)
        perceptron_machine = MLJBase.machine(perceptron, X, y)
        MLJBase.fit!(perceptron_machine)
        ŷ = predict(perceptron_machine, X)
        @test length(ŷ)==length(y)
    end

end
