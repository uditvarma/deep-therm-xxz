# Sanity checks for DeepTherm.  Run with:  julia --project=. sanity_checks.jl
include("src/DeepTherm.jl")
using .DeepTherm, LinearAlgebra, Random, Printf
const DT = DeepTherm

rng = MersenneTwister(1)
ok(name, cond, val) = @printf("%-62s %s   (%.2e)\n", name, cond ? "PASS" : "FAIL", val)

# 1. Hamiltonian: Hermitian, U(1)-symmetric, matches a dense Pauli construction
L = 6; Δ = 0.3
H = Matrix(build_hamiltonian(L; Δ = Δ))
X = [0 1; 1 0]; Y = [0 -im; im 0]; Z = [1 0; 0 -1]; Id = [1 0; 0 1]
op(o, i) = foldl(kron, [j == i ? o : Id for j in L:-1:1])    # site 1 = least significant
Hd = -sum(op(X, i) * op(X, i + 1) + op(Y, i) * op(Y, i + 1) + (1 + Δ) * op(Z, i) * op(Z, i + 1) for i in 1:L-1)
ok("H matches dense Pauli construction", norm(H - Hd) < 1e-12, norm(H - Hd))
Q = sum(op(Z, i) for i in 1:L)
ok("[H, Σ Z_i] = 0", norm(H * Q - Q * H) < 1e-12, norm(H * Q - Q * H))

# 2. Krylov evolution vs dense exponential
L = 8
Hs = build_hamiltonian(L; Δ = 0.5, J2 = 0.3)
ψ0 = random_product_state(rng, L)
t = 7.3
ψK = DT.evolve_measure(Hs, ψ0, [1.0, t], identity)[end]
ψE = exp(-im * t * Matrix(Hs)) * ψ0
ok("Krylov vs dense exp(-iHt), t = 7.3", norm(ψK - ψE) < 1e-9, norm(ψK - ψE))

# 3. Scrooge moments
d = 4
for k in 2:4
    S = DT.perm_sum(d, k)
    haar = S / prod(d:d+k-1)
    scr = DT.scrooge_moment(Matrix{ComplexF64}(I, d, d) / d, k, S)
    ok("Scrooge[1/d]^($k) = Haar^($k)", norm(scr - haar) < 1e-8, norm(scr - haar))
end
# random full-rank ρ: trace, first-moment consistency, and Monte Carlo (GAP sampling)
G = randn(rng, ComplexF64, d, d); ρ = G * G'; ρ /= tr(ρ)
for k in 2:3
    S = DT.perm_sum(d, k)
    scr = DT.scrooge_moment(ρ, k, S)
    ok("tr Scrooge[ρ]^($k) = 1", abs(tr(scr) - 1) < 1e-8, abs(tr(scr) - 1))
    # partial trace over the last k-1 copies should give ρ
    R = reshape(scr, d^(k - 1), d, d^(k - 1), d)       # first kron factor is the slow index
    ρ1 = [sum(R[m, a, m, b] for m in 1:d^(k-1)) for a in 1:d, b in 1:d]
    ok("Tr_{2..k} Scrooge[ρ]^($k) = ρ", norm(ρ1 - ρ) < 1e-8, norm(ρ1 - ρ))
    # Monte Carlo: E[(hh†)^{⊗k}/|h|^{2(k-1)}], h = √ρ g
    sq = sqrt(Hermitian(ρ))
    N = 400_000
    mc = zeros(ComplexF64, d^k, d^k)
    for _ in 1:N
        h = sq * randn(rng, ComplexF64, d)
        v = DT.kron_power(h, k)
        mc .+= (v * v') / norm(h)^(2(k - 1))
    end
    mc ./= N
    td = DT.tracedist(mc, scr)
    ok("Scrooge[ρ]^($k) exact vs Monte Carlo (N=4e5, expect ~1e-3)", td < 1e-2, td)
end

# 4. Benchmark: Haar-random global state on L=10, LA=2
cache = moment_cache(4, 2:4)
vals = [measure_all(haar_state(rng, 10), 2, 10, cache) for _ in 1:20]
for key in (:D1, :D2_haar, :D2_scr, :D3_haar, :D4_haar, :F2, :F4)
    @printf("Haar global state (L=10, LA=2): mean %-8s = %.4f\n", key, sum(getfield.(vals, key)) / 20)
end
