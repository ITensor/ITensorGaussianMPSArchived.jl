using ITensorGaussianMPS
using LinearAlgebra
using ITensors

# Electrons

# Half filling
Nx, Ny = 6, 3
N = Nx * Ny
Nf = N
Nf_up = N ÷ 2
Nf_dn = N - Nf_up

@show Nx, Ny
@show N, Nf

# Maximum MPS link dimension
_maxlinkdim = 1_000

@show _maxlinkdim

# DMRG cutoff
_cutoff = 1e-5

# Hopping
t = 1.0

# Electron-electon on-site interaction
U = 4.0

@show t, U

lattice = square_lattice(Nx, Ny; yperiodic=true)

# Make the free fermion Hamiltonian for the up spins
ampo_up = AutoMPO()
for b in lattice
  ampo_up .+= -t, "Cdagup", b.s1, "Cup", b.s2
  ampo_up .+= -t, "Cdagup", b.s2, "Cup", b.s1
end

# Make the free fermion Hamiltonian for the down spins
ampo_dn = AutoMPO()
for b in lattice
  ampo_dn .+= -t, "Cdagdn", b.s1, "Cdn", b.s2
  ampo_dn .+= -t, "Cdagdn", b.s2, "Cdn", b.s1
end

# Hopping Hamiltonian with 2*N spinless fermions,
# alternating up and down spins
h_up = hopping_hamiltonian(ampo_up)
h_dn = hopping_hamiltonian(ampo_dn)

# Get the Slater determinant
Φ_up = slater_determinant_matrix(h_up, Nf_up)
Φ_dn = slater_determinant_matrix(h_dn, Nf_dn)

println()
println("Exact free fermion energy: ", tr(Φ_up'h_up * Φ_up) + tr(Φ_dn'h_dn * Φ_dn))
println()

# Create an MPS from the slater determinant.
# For now it only works without Sz conservation, this will be supported soon.
s = siteinds("Electron", N; conserve_qns=true)
println("Making free fermion starting MPS")
@time ψ0 = slater_determinant_to_mps(
  s, Φ_up, Φ_dn; eigval_cutoff=1e-4, cutoff=_cutoff, maxdim=_maxlinkdim
)
@show maxlinkdim(ψ0)

ampo = AutoMPO()
for b in lattice
  ampo .+= -t, "Cdagup", b.s1, "Cup", b.s2
  ampo .+= -t, "Cdagdn", b.s1, "Cdn", b.s2
  ampo .+= -t, "Cdagup", b.s2, "Cup", b.s1
  ampo .+= -t, "Cdagdn", b.s2, "Cdn", b.s1
end
for n in 1:N
  ampo .+= U, "Nupdn", n
end
H = MPO(ampo, s)

# Random starting state
ψr = randomMPS(s, n -> n ≤ Nf ? (isodd(n) ? "↑" : "↓") : "0")

println("\nRandom starting state energy")
@show flux(ψr)
@show inner(ψr, H, ψr)

println("\nFree fermion MPS starting state energy")
@show flux(ψ0)
@show inner(ψ0, H, ψ0)

println("\nStart from random product state")
sweeps = Sweeps(10)
maxdim!(sweeps, 10, 20, 100, 200, _maxlinkdim)
cutoff!(sweeps, _cutoff)
noise!(sweeps, 1e-7, 1e-8, 1e-10, 0.0)
@time dmrg(H, ψr, sweeps)

println("\nStart from free fermion state")
sweeps = Sweeps(10)
maxdim!(sweeps, _maxlinkdim)
cutoff!(sweeps, _cutoff)
@time dmrg(H, ψ0, sweeps)

nothing
