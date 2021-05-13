using ITensorGaussianMPS
using ITensors
using LinearAlgebra
using Test

@testset "Electron" begin
  # Half filling
  N = 40
  Nf_up = N ÷ 2
  Nf_dn = N ÷ 2
  Nf = Nf_up + Nf_dn

  # Maximum MPS link dimension
  _maxlinkdim = 200

  # DMRG cutoff
  _cutoff = 1e-8

  # Hopping
  t = 1.0

  # Electron-electron on-site interaction
  U = 1.0

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
  ψ0 = slater_determinant_to_mps(
    s, Φ_up, Φ_dn; eigval_cutoff=1e-4, cutoff=_cutoff, maxdim=_maxlinkdim
  )
  @test maxlinkdim(ψ0) ≤ _maxlinkdim

  # The total non-interacting part of the Hamiltonian
  ampo_noninteracting = AutoMPO()
  for n in 1:(N - 1)
    ampo_noninteracting .+= -t, "Cdagup", n, "Cup", n + 1
    ampo_noninteracting .+= -t, "Cdagdn", n, "Cdn", n + 1
    ampo_noninteracting .+= -t, "Cdagup", n + 1, "Cup", n
    ampo_noninteracting .+= -t, "Cdagdn", n + 1, "Cdn", n
  end

  H_noninteracting = MPO(ampo_noninteracting, s)
  @test tr(Φ_up' * h_up * Φ_up) + tr(Φ_dn' * h_dn * Φ_dn) ≈ inner(ψ0, H_noninteracting, ψ0) rtol =
    1e-3

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

  # Random starting state
  ψr = randomMPS(s, n -> n ≤ Nf ? (isodd(n) ? "↑" : "↓") : "0")

  @test flux(ψr) == QN(("Nf", Nf, -1), ("Sz", 0))
  @test flux(ψ0) == QN(("Nf", Nf, -1), ("Sz", 0))

  @test inner(ψ0, H, ψ0) < inner(ψr, H, ψr)

  sweeps = Sweeps(3)
  maxdim!(sweeps, 10, 20, _maxlinkdim)
  cutoff!(sweeps, _cutoff)
  noise!(sweeps, 1e-5, 1e-6, 1e-7, 0.0)
  er, _ = dmrg(H, ψr, sweeps; outputlevel=0)

  sweeps = Sweeps(3)
  maxdim!(sweeps, _maxlinkdim)
  cutoff!(sweeps, _cutoff)
  noise!(sweeps, 1e-5, 1e-6, 1e-7, 0.0)
  e0, _ = dmrg(H, ψ0, sweeps; outputlevel=0)

  @test e0 > inner(ψ0, H_noninteracting, ψ0)
  @test e0 < er
end
