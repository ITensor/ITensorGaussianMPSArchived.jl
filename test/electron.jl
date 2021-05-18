using ITensorGaussianMPS
using ITensors
using LinearAlgebra
using Test

function expect_compat(psi::MPS, ops::AbstractString...; kwargs...)
  if ITensors.version() >= v"0.2"
    return expect(psi, ops...; kwargs...)
  end
  psi = copy(psi)
  N = length(psi)
  ElT = real(promote_itensor_eltype(psi))
  Nops = length(ops)
  s = siteinds(psi)
  site_range::UnitRange{Int} = get(kwargs, :site_range, 1:N)
  Ns = length(site_range)
  start_site = first(site_range)
  offset = start_site - 1
  orthogonalize!(psi, start_site)
  psi[start_site] ./= norm(psi[start_site])
  ex = ntuple(n -> zeros(ElT, Ns), Nops)
  for j in site_range
    orthogonalize!(psi, j)
    for n in 1:Nops
      ex[n][j - offset] = real(scalar(psi[j] * op(ops[n], s[j]) * dag(prime(psi[j], s[j]))))
    end
  end
  return Nops == 1 ? ex[1] : ex
end

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

@testset "Regression test for bug away from half filling" begin
  N = 3
  t = 1.0
  ampo_up = AutoMPO()
  for n in 1:(N - 1)
    ampo_up .+= -t, "Cdagup", n, "Cup", n + 1
    ampo_up .+= -t, "Cdagup", n + 1, "Cup", n
  end
  ampo_dn = AutoMPO()
  for n in 1:(N - 1)
    ampo_dn .+= -t, "Cdagdn", n, "Cdn", n + 1
    ampo_dn .+= -t, "Cdagdn", n + 1, "Cdn", n
  end
  h_up = hopping_hamiltonian(ampo_up)
  h_dn = hopping_hamiltonian(ampo_dn)
  s = siteinds("Electron", N; conserve_qns=true)
  H = MPO(ampo_up + ampo_dn, s)
  Nf_up, Nf_dn = 1, 0
  Φ_up = slater_determinant_matrix(h_up, Nf_up)
  Φ_dn = slater_determinant_matrix(h_dn, Nf_dn)
  ψ = slater_determinant_to_mps(s, Φ_up, Φ_dn; eigval_cutoff=0.0, cutoff=0.0)
  @test inner(ψ, H, ψ) ≈ tr(Φ_up'h_up * Φ_up) + tr(Φ_dn'h_dn * Φ_dn)
  @test maxlinkdim(ψ) == 2
  @test flux(ψ) == QN(("Nf", 1, -1), ("Sz", 1))
  ns_up = expect_compat(ψ, "Nup")
  ns_dn = expect_compat(ψ, "Ndn")
  @test ns_up ≈ diag(Φ_up * Φ_up')
  @test ns_dn ≈ diag(Φ_dn * Φ_dn')
  @test sum(ns_up) ≈ Nf_up
  @test sum(ns_dn) ≈ Nf_dn
end
