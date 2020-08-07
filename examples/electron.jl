using GaussianMatrixProductStates
using LinearAlgebra
using ITensors

# Electrons

# Half filling
N = 20
Nf = N

@show N, Nf


# Hopping
t = 1.0

h = Hermitian(diagm(2 => fill(-t, 2*N-2), -2 => fill(-t, 2*N-2)))

_, u = eigen(h)

# Get the Slater determinant
Φ = u[:, 1:Nf]

# Create an mps

# Fermion sites will be combined into electron sites
sf = siteinds("Fermion", 2*N; conserve_qns = true)
s = siteinds("Electron", N; conserve_qns = true, conserve_sz = false)

ψ0f = slater_determinant_to_mps(sf, Φ; blocksize = 6)
ψ0 = MPS(N)
for n in 1:N
  i, j = 2*n-1, 2*n
  C = combiner(sf[i], sf[j])
  c = combinedind(C)
  ψ0[n] = ψ0f[i] * ψ0f[j] * C
  ψ0[n] *= δ(dag(c), s[n])
end

U = 1.0
@show U
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

# Random tarting state
ψr = randomMPS(s, n -> n ≤ Nf ? (isodd(n) ? "↑" : "↓") : "0")

@show inner(ψr, H, ψr)
@show inner(ψ0, H, ψ0)

println("\nStart from product state")
sweeps = Sweeps(10)
maxdim!(sweeps,10,20,100)
cutoff!(sweeps,1E-12)
dmrg(H, ψr, sweeps)

println("\nStart from free fermion state")
sweeps = Sweeps(3)
maxdim!(sweeps,100)
cutoff!(sweeps,1E-12)
dmrg(H, ψ0, sweeps)

nothing