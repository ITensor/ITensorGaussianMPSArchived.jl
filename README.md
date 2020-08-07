# GaussianMatrixProductStates

A package for creating the matrix product state of a free fermion (Gaussian) state.

## Installation

The package is currently unregistered. To install it, start `julia` and run the command:
```julia
julia>]

pkg> add https://github.com/mtfishman/GaussianMatrixProductStates.jl
```

## Examples

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
This will output something like:
```julia
(N, Nf) = (20, 10)
U = 1.0

Free fermion starting energy
inner(ψ0, H, ψ0) = -2.3812770621299357

Random state starting energy
inner(ψr, H, ψr) = 10.0

Run dmrg with random starting state
After sweep 1 energy=6.261701784151 maxlinkdim=2 time=0.041
After sweep 2 energy=2.844954346204 maxlinkdim=5 time=0.056
After sweep 3 energy=0.245282430911 maxlinkdim=14 time=0.071
After sweep 4 energy=-1.439072132586 maxlinkdim=32 time=0.098
After sweep 5 energy=-2.220202191945 maxlinkdim=59 time=0.148
After sweep 6 energy=-2.376787647893 maxlinkdim=60 time=0.186
After sweep 7 energy=-2.381484153892 maxlinkdim=60 time=0.167
After sweep 8 energy=-2.381489999291 maxlinkdim=57 time=0.233
After sweep 9 energy=-2.381489999595 maxlinkdim=49 time=0.175
After sweep 10 energy=-2.381489999595 maxlinkdim=49 time=0.172
  1.349192 seconds (8.94 M allocations: 1.027 GiB, 18.05% gc time)

Run dmrg with free fermion starting state
After sweep 1 energy=-2.381489929965 maxlinkdim=49 time=0.139
After sweep 2 energy=-2.381489999588 maxlinkdim=49 time=0.165
After sweep 3 energy=-2.381489999594 maxlinkdim=48 time=0.161
After sweep 4 energy=-2.381489999594 maxlinkdim=48 time=0.169
  0.637021 seconds (4.59 M allocations: 525.989 MiB, 17.09% gc time)
```
