using ITensorGaussianMPS
using LinearAlgebra
using ITensors

# Electrons

# Half filling
N = 100
Nf_up = N ÷ 2
Nf_dn = N ÷ 2
Nf = Nf_up + Nf_dn

@show N, Nf

# Maximum MPS link dimension
_maxlinkdim = 200

@show _maxlinkdim

# DMRG cutoff
_cutoff = 1e-8

# Hopping
t = 1.0

# Electron-electron on-site interaction
U = 1.0

@show t, U

# Make the free fermion Hamiltonian for the up spins
ampo_up = AutoMPO()
for n in 1:(N - 1)
  ampo_up .+= -t, "Cdagup", n, "Cup", n + 1
  ampo_up .+= -t, "Cdagup", n + 1, "Cup", n
end

# Make the free fermion Hamiltonian for the down spins
ampo_dn = AutoMPO()
for n in 1:(N - 1)
  ampo_dn .+= -t, "Cdagdn", n, "Cdn", n + 1
  ampo_dn .+= -t, "Cdagdn", n + 1, "Cdn", n
end

# Hopping Hamiltonians for the up and down spins
h_up = hopping_hamiltonian(ampo_up)
h_dn = hopping_hamiltonian(ampo_dn)

# Get the Slater determinant
Φ_up = slater_determinant_matrix(h_up, Nf_up)
Φ_dn = slater_determinant_matrix(h_dn, Nf_dn)

# Create an MPS from the slater determinants.
s = siteinds("Electron", N; conserve_qns=true)
println("Making free fermion starting MPS")
@time ψ0 = slater_determinant_to_mps(
  s, Φ_up, Φ_dn; eigval_cutoff=1e-4, cutoff=_cutoff, maxdim=_maxlinkdim
)
@show maxlinkdim(ψ0)

# The total non-interacting part of the Hamiltonian
ampo_noninteracting = AutoMPO()
for n in 1:(N - 1)
  ampo_noninteracting .+= -t, "Cdagup", n, "Cup", n + 1
  ampo_noninteracting .+= -t, "Cdagdn", n, "Cdn", n + 1
  ampo_noninteracting .+= -t, "Cdagup", n + 1, "Cup", n
  ampo_noninteracting .+= -t, "Cdagdn", n + 1, "Cdn", n
end

H_noninteracting = MPO(ampo_noninteracting, s)
@show inner(ψ0, H_noninteracting, ψ0)
@show sum(diag(Φ_up' * h_up * Φ_up)) + sum(diag(Φ_dn' * h_dn * Φ_dn))

# The total interacting Hamiltonian
ampo_interacting = AutoMPO()
for n in 1:(N - 1)
  ampo_interacting .+= -t, "Cdagup", n, "Cup", n + 1
  ampo_interacting .+= -t, "Cdagdn", n, "Cdn", n + 1
  ampo_interacting .+= -t, "Cdagup", n + 1, "Cup", n
  ampo_interacting .+= -t, "Cdagdn", n + 1, "Cdn", n
end
for n in 1:N
  ampo_interacting .+= U, "Nupdn", n
end
H = MPO(ampo_interacting, s)
#@show norm(prod(H) - prod(H_noninteracting))

# Random starting state
ψr = randomMPS(s, n -> n ≤ Nf ? (isodd(n) ? "↑" : "↓") : "0")

println("Random starting state energy")
@show flux(ψr)
@show inner(ψr, H, ψr)
println()
println("Free fermion starting state energy")
@show flux(ψ0)
@show inner(ψ0, H, ψ0)

println("\nStart from random product state")
sweeps = Sweeps(10)
maxdim!(sweeps, 10, 20, _maxlinkdim)
cutoff!(sweeps, _cutoff)
er, ψ̃r = @time dmrg(H, ψr, sweeps)
@show er
@show flux(ψ̃r)

println("\nStart from free fermion state")
sweeps = Sweeps(5)
maxdim!(sweeps, _maxlinkdim)
cutoff!(sweeps, _cutoff)
e0, ψ̃0 = @time dmrg(H, ψ0, sweeps)
@show e0
@show flux(ψ̃0)

nothing
