include("GaussianMatrixProductStates.jl")

# Electrons

N = 10
Nf = N

s = siteinds("Electron", N; conserve_qns = true, conserve_sz = false)

# Starting state
ψ0 = productMPS(s, n -> n ≤ Nf ? (isodd(n) ? "↑" : "↓") : "0")

# Hopping
t = 1.0

ampo = AutoMPO()
for b=1:N-1
  ampo .+= -t,"Cdagup",b,"Cup",b+1
  ampo .+= -t,"Cdagup",b+1,"Cup",b
  ampo .+= -t,"Cdagdn",b,"Cdn",b+1
  ampo .+= -t,"Cdagdn",b+1,"Cdn",b
end
H = MPO(ampo, s)

sweeps = Sweeps(10)
maxdim!(sweeps,10,20,40,60)
cutoff!(sweeps,1E-12)
energy, ψ = dmrg(H, ψ0, sweeps)

h = Hermitian(diagm(2 => fill(-t, 2*N-2), -2 => fill(-t, 2*N-2)))

e, u = eigen(h)
d = Diagonal(e)
@assert h * u ≈ u * d

@show sum(e[1:Nf]), energy

# Get the Slater determinant
Φ = u[:, 1:Nf]
d_Nf = d[1:Nf, 1:Nf]

@assert h * Φ ≈ Φ * d_Nf

# Create an mps

sf = siteinds("Fermion", 2*N; conserve_qns = true)
ψ̃f = slater_determinant_to_mps(sf, Φ; blocksize = 6)
ψ̃ = MPS(N)
for n in 1:N
  i, j = 2*n-1, 2*n
  C = combiner(sf[i], sf[j])
  c = combinedind(C)
  ψ̃[n] = ψ̃f[i] * ψ̃f[j] * C
  ψ̃[n] *= δ(dag(c), s[n])
end
@show maxlinkdim(ψ), maxlinkdim(ψ̃)
@show inner(ψ, ψ̃)
@show inner(ψ̃, H, ψ̃), inner(ψ, H, ψ)

U = 1.0
ampo = AutoMPO()
for b=1:N-1
  ampo .+= -t,"Cdagup",b,"Cup",b+1
  ampo .+= -t,"Cdagup",b+1,"Cup",b
  ampo .+= -t,"Cdagdn",b,"Cdn",b+1
  ampo .+= -t,"Cdagdn",b+1,"Cdn",b
end
for i in 1:N
  ampo .+= U, "Nupdn", i
end
H = MPO(ampo, s)

println("\n Start from product state")
sweeps = Sweeps(5)
maxdim!(sweeps,10,20,100)
cutoff!(sweeps,1E-12)
dmrg(H, ψ0, sweeps)

println("\n Start from free fermion state")
sweeps = Sweeps(5)
maxdim!(sweeps,100)
cutoff!(sweeps,1E-12)
dmrg(H, ψ̃, sweeps)