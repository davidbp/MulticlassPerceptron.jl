using Random

export make_blobs

uniform_sample_in_zero_maxval(n_features, maxval) = maxval .* (1 .- rand(n_features)) 

uniform_sample_in_minval_maxval(n_features, minval, maxval) = (maxval-minval) .* rand(n_features) .+ minval 



"""
    make_blobs(n_examples=100; 
               n_features=2, 
               centers=3, 
               cluster_std=1.0, 
               center_box=(-10.,10.),
               element_type=Float64,
               random_seed=1234,
               return_centers=false)
               
     Generates a dataset with `n_examples` of dimension `n_features` and returns a vector containing
     as integers the membership of the different points generated.

     The data is roughly grouped around several `centers`  which are created using `cluster_std`. 
     The data lives inside `center_box` in the case it is randomly generated.
     
     - If `centers` is an integer the centroids are created randomly.    
     - If `centers` is an Array containing points the centroids are picked from `centers`.
     - If `return_centers=true` the centroids of the bloods are returned

"""
function make_blobs(n_examples=100; n_features=2, 
                    centers=3, cluster_std=1.0, center_box=(-10.,10.),
                    element_type=Float64, random_seed=1234, return_centers=false)

    Random.seed!(random_seed)

    X = [] 
    y = []

    if typeof(centers)<: Int
        n_centers = centers
        centers = []
        for c in 1:n_centers
            center_sample = uniform_sample_in_minval_maxval(n_features, center_box[1], center_box[2])
            push!(centers, center_sample)
        end
    else
        n_centers = length(centers)
    end

    if typeof(cluster_std)<: AbstractFloat
        cluster_std = 0.5 * randn(n_centers)
    end
    
    # generates the nunber of example for each center
    n_examples_per_center = [div(n_examples, n_centers) for x  in 1:n_centers]
    for i in 1:(n_examples % n_centers)
        n_examples_per_center[i] += 1
    end

    # generates the actual vectors close to each center blob 
    for (i, (n, std, center)) in enumerate(zip(n_examples_per_center, cluster_std, centers))
        X_current = center' .+ std .* randn(element_type, (n, n_features))
        push!(X, X_current)
        push!(y, [i for k in 1:n]) 
    end

    # stack all the previous arrays created for each of the centers
    X = cat(X..., dims=1)
    y = cat(y..., dims=1)
    
    if return_centers
        return X, y, centers  
    else
        return X, y
    end
end

