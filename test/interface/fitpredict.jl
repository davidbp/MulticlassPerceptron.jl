
# 10*(1-rand()) is uniform from (0, 10]
uniform_sample_in_range(n_features, minval, maxval) = 2*maxval*rand(n_features) .+ minval

# x=rand() is a sample from N(0,1) =>  x2 = mu + sqrt(std)*rand() is a sample from  N(mu, std)
normal_sample(n_features, mu, var) = mu .+ sqrt(var) .* randn(n_features)

# make bloobs
function make_blobs(n_examples=100, n_features=2, n_classes=3, 
                    centers=3, cluster_std=1.0, center_box=(-10.,10.),
                    element_type=Float64, random_seed=1234)

    Random.seed!(random_seed)

    X = [] 
    y = []

    if typeof(centers)<: Int
        n_centers = centers
        centers = []
        for c in 1:n_centers
            center_sample = uniform_sample_in_range(center_box[1],center_box[1])
            push!(centers, center_sample)
        end
    end

    # generate the nunber of example for each center
    n_examples_per_center = [div(n_examples, n_classes) for x  in 1:n_centers]
    for i in 1:(n_examples % n_centers)
        n_examples_per_center[i] += 1
    end

    # Generate the actual vectors close to each center bloob 
    for (i, (n, std, center)) in enumerate(zip(n_examples_per_center, cluster_std, centers))
        X_current = center .+ cluster_std .* randn(element_type, (n, n_features))
        push!(X, X_current)
        push!(y, [i for i in 1:n]) 
    end

    # get a single Array that stacks all the previous arrays created for each of the centers
    X = cat(X..., dims=1)
    return X, y
end

@testset "multiclass_perceptron_fit_2_bloobs" begin
    Random.seed!(6161)
    n, p = 100, 2
    n_classes = 2
    X = randn(n, p)
    y = rand(1:n_classes, n, 10)
    
    Xt = MLJBase.table(X)
    percep = MulticlassPerceptron()
    fr, = MLJBase.fit(percep, 1, Xt, y)
    ŷ   = MLJBase.predict(rr, fr, Xt)

    coefs = θ[1:end-1]
    intercept = θ[end]

    fp = MLJBase.fitted_params(rr, fr)
    @test fp.coefs ≈ coefs
    @test fp.intercept ≈ intercept
end


@testset "multiclass_perceptron_fit_3_bloobs" begin
    Random.seed!(6161)
    n, p = 100, 2
    n_classes = 3
    X = randn(n, p)
    y = rand(1:n_classes, n, 10)
    
    Xt = MLJBase.table(X)
    percep = MulticlassPerceptron()
    fr, = MLJBase.fit(percep, 1, Xt, y)
    ŷ   = MLJBase.predict(rr, fr, Xt)

    coefs = θ[1:end-1]
    intercept = θ[end]

    fp = MLJBase.fitted_params(rr, fr)
    @test fp.coefs ≈ coefs
    @test fp.intercept ≈ intercept
end