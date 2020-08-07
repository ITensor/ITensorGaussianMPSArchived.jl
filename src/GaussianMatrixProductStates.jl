module GaussianMatrixProductStates

using ITensors
using LinearAlgebra

import LinearAlgebra: Givens

export slater_determinant_to_mps,
       slater_determinant_to_gmps

include("gmps.jl")

end
