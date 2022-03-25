

function load_MNIST_flux( ;array_eltype::DataType=Float32, verbose::Bool=true)

    if verbose
        time_init = time()
        println("MNIST Dataset Loading...")
    end
    train_imgs = MNIST.images(:train)                             # size(train_imgs) -> (60000,)
    test_imgs  = MNIST.images(:test)                              # size(test_imgs)  -> (10000,)
    train_x    = array_eltype.(hcat(reshape.(train_imgs, :)...))  # size(train_x)    -> (784, 60000)
    test_x     = array_eltype.(hcat(reshape.(test_imgs, :)...))   # size(test_x)     -> (784, 60000)

    ## Prepare data
    train_y = MNIST.labels(:train) .+ 1;
    test_y  = MNIST.labels(:test)  .+ 1;

    ## CategoricalArray are not needed for the MulticlassPerceptronCore 
    #train_y = CategoricalArray(train_y)
    #test_y  = CategoricalArray(test_y)

    if verbose
        time_taken = round(time()-time_init; digits=3)
        println("MNIST Dataset Loaded, it took $time_taken seconds")
    end
    return train_x, train_y, test_x, test_y
end