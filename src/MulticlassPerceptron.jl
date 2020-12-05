module MulticlassPerceptron 

# MulticlassPerceptronCore
include("MulticlassPerceptronCore.jl")

# MLJ interface
include("mlj/interface.jl")

# Synthetic Data generator
include("synthetic_data/synthetic_data_generation.jl")

end # end module