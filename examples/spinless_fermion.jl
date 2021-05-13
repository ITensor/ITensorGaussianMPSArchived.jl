using ITensorGaussianMPS
using LinearAlgebra
using ITensors

# Half filling
N = 50
Nf = N ÷ 2

@show N, Nf

# Maximum MPS link dimension
_maxlinkdim = 100

@show _maxlinkdim

# DMRG cutoff
_cutoff = 1e-12

# Hopping
t = 1.0

# Electron-electron on-site interaction
U = 1.0

@show t, U

# Free fermion Hamiltonian
ampo = AutoMPO()
for n in 1:(N - 1)
  ampo .+= -t, "Cdag", n, "C", n + 1
  ampo .+= -t, "Cdag", n + 1, "C", n
end

# Hopping Hamiltonian with N spinless fermions
h = hopping_hamiltonian(ampo)

# Get the Slater determinant
Φ = slater_determinant_matrix(h, Nf)

# Create an mps for the free fermion ground state
s = siteinds("Fermion", N; conserve_qns=true)
println("Making free fermion starting MPS")
@time ψ0 = slater_determinant_to_mps(
  s, Φ; eigval_cutoff=1e-4, cutoff=_cutoff, maxdim=_maxlinkdim
)
@show maxlinkdim(ψ0)

# Make an interacting Hamiltonian
for n in 1:(N - 1)
  ampo .+= U, "N", n, "N", n + 1
end
H = MPO(ampo, s)

# Random starting state
ψr = randomMPS(s, n -> n ≤ Nf ? "1" : "0")

println("\nRandom state starting energy")
@show flux(ψr)
@show inner(ψr, H, ψr)

println("\nFree fermion starting energy")
@show flux(ψ0)
@show inner(ψ0, H, ψ0)

println("\nRun dmrg with random starting state")
sweeps = Sweeps(20)
maxdim!(sweeps, 10, 20, 40, _maxlinkdim)
cutoff!(sweeps, _cutoff)
@time dmrg(H, ψr, sweeps)

println("\nRun dmrg with free fermion starting state")
sweeps = Sweeps(4)
maxdim!(sweeps, _maxlinkdim)
cutoff!(sweeps, _cutoff)
@time dmrg(H, ψ0, sweeps)

nothing
