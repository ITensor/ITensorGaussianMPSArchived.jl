# GaussianMatrixProductStates

A package for creating the matrix product state of a free fermion (Gaussian) state.

This can help create starting states for DMRG. For example:
```julia
using ITensors
using GaussianMatrixProductStates

# Half filling
N = 20
Nf = N÷2

@show N, Nf

# Hopping
t = 1.0

# Free fermion hopping Hamiltonian
#h = SymTridiagonal(zeros(N), fill(-t, N-1))
h = Hermitian(diagm(1 => fill(-t, N-1), -1 => fill(-t, N-1)))
_, u = eigen(h)

# Get the Slater determinant
Φ = u[:, 1:Nf]

# Create an mps for the free fermion ground state
s = siteinds("Fermion", N; conserve_qns = true)
ψ0 = slater_determinant_to_mps(s, Φ; blocksize = 4)

# Make an interacting Hamiltonian
U = 1.0
@show U

ampo = AutoMPO()
for b in 1:N-1
  ampo .+= -t,"Cdag",b,"C",b+1
  ampo .+= -t,"Cdag",b+1,"C",b
end
for b in 1:N
  ampo .+= U, "Cdag*C", b
end
H = MPO(ampo, s)

println("\nFree fermion starting energy")
@show inner(ψ0, H, ψ0)

# Random starting state
ψr = randomMPS(s, n -> n ≤ Nf ? "1" : "0")

println("\nRandom state starting energy")
@show inner(ψr, H, ψr)

println("\nRun dmrg with random starting state")
sweeps = Sweeps(10)
maxdim!(sweeps,10,20,40,60)
cutoff!(sweeps,1E-12)
@time dmrg(H, ψr, sweeps)

println("\nRun dmrg with free fermion starting state")
sweeps = Sweeps(4)
maxdim!(sweeps,60)
cutoff!(sweeps,1E-12)
@time dmrg(H, ψ0, sweeps)
```
