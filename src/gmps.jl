#
# Single particle von Neumann entanglement entropy
#

function entropy(n::Number)
  (n ≤ 0 || n ≥ 1) && return 0
  return -(n*log(n) + (1-n)*log(1-n))
end

entropy(ns::Vector{Float64}) = sum(entropy, ns)

#
# Linear Algebra tools
#

"""
    frobenius_distance(M1::AbstractMatrix, M2::AbstractMatrix)

Computes the Frobenius distance `√tr((M1-M2)'*(M1-M2))`.
"""
frobenius_distance(M1::AbstractMatrix, M2::AbstractMatrix) =
  sqrt(abs(tr(M1'M1) + tr(M2'M2) - tr(M1'M2) - tr(M2'M1)))

#
# Rotations
#

struct Circuit{T} <: LinearAlgebra.AbstractRotation{T}
  rotations::Vector{Givens{T}}
end

Base.adjoint(R::Circuit) = Adjoint(R)

Base.copy(aR::Adjoint{<:Any,Circuit{T}}) where {T} = Circuit{T}(reverse!([r' for r in aR.parent.rotations]))

function LinearAlgebra.lmul!(G::Givens, R::Circuit)
  push!(R.rotations, G)
  return R
end

function LinearAlgebra.lmul!(R::Circuit, A::AbstractArray)
  @inbounds for i = 1:length(R.rotations)
      lmul!(R.rotations[i], A)
  end
  return A
end

function LinearAlgebra.rmul!(A::AbstractMatrix, adjR::Adjoint{<:Any,<:Circuit})
  R = adjR.parent
  @inbounds for i = 1:length(R.rotations)
      rmul!(A, adjoint(R.rotations[i]))
  end
  return A
end

Base.:*(g1::Circuit, g2::Circuit) = Circuit(vcat(g2.rotations, g1.rotations))
LinearAlgebra.lmul!(g1::Circuit, g2::Circuit) = append!(g2.rotations, g1.rotations)

Base.:*(A::Circuit, B::Union{ <: Hermitian, <: Diagonal}) = A * convert(Matrix, B)
Base.:*(A::Adjoint{ <: Any, <: Circuit}, B::Hermitian) = copy(A) * convert(Matrix, B)
Base.:*(A::Adjoint{ <: Any, <: Circuit}, B::Diagonal) = copy(A) * convert(Matrix, B)
Base.:*(A::Adjoint{<:Any, <: AbstractVector}, B::Adjoint{ <:Any, <: Circuit}) = convert(Matrix, A) * B

function LinearAlgebra.rmul!(A::AbstractMatrix, R::Circuit)
  @inbounds for i = reverse(1:length(R.rotations))
      rmul!(A, R.rotations[i])
  end
  return A
end

function shift!(G::Circuit, i::Int)
  for (n, g) in enumerate(G.rotations)
    G.rotations[n] = Givens(g.i1+i, g.i2+i, g.c, g.s)
  end
  return G
end

ngates(G::Circuit) = length(G.rotations)

#
# Correlation matrix diagonalization
#

"""
    givens_rotations(v::AbstractVector)

For a vector `v`, return the `length(v)-1`
Givens rotations `g` and the norm `r` such that:
```julia
g * v ≈ r * [n == 1 ? 1 : 0 for n in 1:length(v)]
```
"""
function givens_rotations(v::AbstractVector{ElT}) where {ElT}
  N = length(v)
  gs = Circuit{ElT}([])
  r = v[1]
  for n in reverse(1:N-1)
    g, r = givens(v, n, n+1)
    v = g * v
    lmul!(g, gs)
  end
  return gs, r
end

"""
    correlation_matrix_to_gmps(Λ::AbstractMatrix{ElT}; blocksize::Int)

Diagonalize a correlation matrix, returning the eigenvalues and eigenvectors
stored in a structure as a set of Givens rotations.

The correlation matrix should be Hermitian, and will be treated as if it itensor
in the algorithm.
"""
function correlation_matrix_to_gmps(Λ0::AbstractMatrix{ElT}; blocksize::Int) where {ElT <: Number}
  Λ = Hermitian(Λ0)
  N = size(Λ, 1)
  V = Circuit{ElT}([])
  ns = Vector{ElT}(undef, N)
  for i in 1:N
    j = min(i + blocksize, N)
    ΛB = @view Λ[i:j, i:j]
    nB, uB = eigen(ΛB)
    p = sortperm(nB; by = entropy)
    n = nB[p[1]]
    ns[i] = n
    v = @view uB[:, p[1]]
    g, _ = givens_rotations(v)
    shift!(g, i-1)
    # In-place version of:
    # V = g * V
    lmul!(g, V)
    Λ = Hermitian(g * Λ * g')
  end
  return ns, V
end

function slater_determinant_to_gmps(Φ::AbstractMatrix; kwargs...)
  return correlation_matrix_to_gmps(conj(Φ) * transpose(Φ); kwargs...)
end

#
# Turn circuit into MPS
#

function ITensors.ITensor(u::Givens, s1::Index, s2::Index)
  U = [1    0   0 0
       0  u.c u.s 0
       0 -u.s u.c 0
       0    0   0 1]
  return itensor(U, s2', s1', dag(s2), dag(s1))
end

function ITensors.ITensor(sites::Vector{<:Index}, u::Givens)

  s1 = sites[u.i1]
  s2 = sites[u.i2]
  return ITensor(u, s1, s2)
end

"""
    MPS(sites::Vector{<:Index}, state, U::Vector{<:ITensor}; kwargs...)

Return an MPS with site indices `sites` by applying the circuit `U` to the starting state `state`.
"""
function ITensors.MPS(sites::Vector{<:Index}, state, U::Vector{<:ITensor}; kwargs...)
  return apply(U..., productMPS(sites, state); kwargs...)
end

"""
    correlation_matrix_to_mps(Λ::AbstractMatrix{ElT}; blocksize::Int, kwargs...)

Return an approximation to the state represented by the correlation matrix as
a matrix product state (MPS).

The correlation matrix should correspond to a pure state (have all eigenvalues
of zero or one).
"""
function correlation_matrix_to_mps(s::Vector{<:Index}, Λ::AbstractMatrix{ElT}; blocksize::Int, kwargs...) where {ElT <: Number}
  ns, C = correlation_matrix_to_gmps(Λ; blocksize)
  U = [ITensor(s, g) for g in C.rotations]
  return MPS(s, n -> round(Int, ns[n]) + 1, U; kwargs...)
end

function slater_determinant_to_mps(s::Vector{<:Index}, Φ::AbstractMatrix; kwargs...)
  return correlation_matrix_to_mps(s, conj(Φ) * transpose(Φ); kwargs...)
end