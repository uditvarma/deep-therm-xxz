"""
    DeepTherm

Projected ensembles of the XXZ chain after a quench from a random product state.

Hamiltonian (Pauli matrices, open boundaries):

    H = -J  Σ_{i=1}^{L-1} [X_i X_{i+1} + Y_i Y_{i+1} + (1+Δ) Z_i Z_{i+1}]
        -J2 Σ_{i=1}^{L-2} [X_i X_{i+2} + Y_i Y_{i+2} + (1+Δ) Z_i Z_{i+2}]   (breaks integrability, keeps U(1))
        + Σ_i h_i Z_i                                                        (optional fields, keep U(1))

Basis convention: index s = 1 + Σ_i b_i 2^{i-1}, with b_i = 0 ↔ |↑⟩ (Z = +1).
Site 1 is the least significant bit, so `reshape(ψ, 2^LA, 2^LB)` gives Ψ[a, z_B]
with subsystem A = sites 1..LA (one end of the chain) and B = sites LA+1..L.

Moments: ρ^(k) = Σ_z p(z) (|ψ_z⟩⟨ψ_z|)^{⊗k}, compared (trace distance) with
  * Haar:               Π_sym / binom(d+k-1, k)
  * Scrooge[ρ_A]:       maximum-entropy ensemble with the same first moment
  * generalized Scrooge Σ_m p(m) Scrooge[ρ_m], m = number of down spins measured in B
                        (only meaningful for z-basis measurement, where m reveals Q_B)
"""
module DeepTherm

using LinearAlgebra, SparseArrays, Random, KrylovKit, QuadGK

export build_hamiltonian, random_product_state, haar_state, rotate_B,
       moment_cache, measure_all, evolve_measure

# ----------------------------------------------------------------------------
# Hamiltonian
# ----------------------------------------------------------------------------
function build_hamiltonian(L::Int; J=1.0, Δ=0.0, J2=0.0, h=zeros(L))
    dim = 2^L
    rows = Int[]; cols = Int[]; vals = Float64[]
    bonds = Tuple{Int,Int,Float64}[(i, i + 1, J) for i in 1:L-1]
    if J2 != 0
        append!(bonds, [(i, i + 2, J2) for i in 1:L-2])
    end
    for s in 0:dim-1
        diag = 0.0
        for i in 1:L
            diag += h[i] * (1 - 2 * ((s >> (i - 1)) & 1))
        end
        for (i, j, c) in bonds
            bi = (s >> (i - 1)) & 1
            bj = (s >> (j - 1)) & 1
            diag += -c * (1 + Δ) * (1 - 2bi) * (1 - 2bj)
            if bi != bj                      # (XX+YY)|01⟩ = 2|10⟩
                s2 = s ⊻ ((1 << (i - 1)) | (1 << (j - 1)))
                push!(rows, s2 + 1); push!(cols, s + 1); push!(vals, -2c)
            end
        end
        push!(rows, s + 1); push!(cols, s + 1); push!(vals, diag)
    end
    return sparse(rows, cols, vals, dim, dim)
end

# ----------------------------------------------------------------------------
# States
# ----------------------------------------------------------------------------
"Product state with each spin drawn uniformly on the Bloch sphere."
function random_product_state(rng::AbstractRNG, L::Int)
    ψ = ComplexF64[1.0]
    for _ in 1:L
        θ = acos(2rand(rng) - 1)
        φ = 2π * rand(rng)
        v = ComplexF64[cos(θ / 2), cis(φ) * sin(θ / 2)]
        ψ = kron(v, ψ)                       # new site = next more significant bit
    end
    return ψ
end

"Haar-random state on the full 2^L Hilbert space (benchmark)."
function haar_state(rng::AbstractRNG, L::Int)
    ψ = randn(rng, ComplexF64, 2^L)
    return ψ / norm(ψ)
end

"Apply a single-site 2×2 unitary U to site i."
function apply_1site(ψ::Vector{ComplexF64}, U::AbstractMatrix, i::Int, L::Int)
    A = reshape(ψ, 2^(i - 1), 2, 2^(L - i))
    out = similar(A)
    @inbounds for c in axes(A, 3), a in axes(A, 1)
        x0 = A[a, 1, c]; x1 = A[a, 2, c]
        out[a, 1, c] = U[1, 1] * x0 + U[1, 2] * x1
        out[a, 2, c] = U[2, 1] * x0 + U[2, 2] * x1
    end
    return vec(out)
end

"""
Rotate the B sites so that a computational-basis readout corresponds to measuring
in `basis` (:z, :x or :y).
"""
function rotate_B(ψ::Vector{ComplexF64}, LA::Int, L::Int, basis::Symbol)
    basis === :z && return ψ
    U = basis === :x ? ComplexF64[1 1; 1 -1] / sqrt(2) :
        basis === :y ? ComplexF64[1 -im; 1 im] / sqrt(2) :
        error("unknown basis $basis")
    for i in LA+1:L
        ψ = apply_1site(ψ, U, i, L)
    end
    return ψ
end

# ----------------------------------------------------------------------------
# Tensor-power helpers
# ----------------------------------------------------------------------------
function all_perms(k::Int)
    k == 1 && return [[1]]
    out = Vector{Vector{Int}}()
    for p in all_perms(k - 1), pos in 1:k
        q = copy(p); insert!(q, pos, k); push!(out, q)
    end
    return out
end

"digits of a 0-based index in base d, most significant first (matches kron ordering), 1-based"
tdigits(idx, d, k) = [(idx ÷ d^(k - j)) % d + 1 for j in 1:k]

"S = Σ_{π∈S_k} P_π on (C^d)^{⊗k}"
function perm_sum(d::Int, k::Int)
    D = d^k
    S = zeros(Float64, D, D)
    perms = all_perms(k)
    for idx in 0:D-1
        dg = tdigits(idx, d, k)
        for π in perms
            pd = dg[π]
            idx2 = sum((pd[j] - 1) * d^(k - j) for j in 1:k)
            S[idx2+1, idx+1] += 1
        end
    end
    return S
end

kron_power(A, k) = k == 1 ? A : kron(A, kron_power(A, k - 1))

"Cache of k-independent objects: permutation sums and Haar moments."
function moment_cache(dA::Int, ks)
    S = Dict(k => perm_sum(dA, k) for k in ks)
    haar = Dict(k => S[k] / prod(dA:dA+k-1) for k in ks)
    return (; dA, ks = collect(ks), S, haar)
end

# ----------------------------------------------------------------------------
# Scrooge ensemble: exact k-th moment
#
# Scrooge[ρ] = GAP measure: |ψ⟩ = h/|h| with h ~ CN(0, ρ), weighted by |h|².
#   ρ^(k) = E[(h h†)^{⊗k} / |h|^{2(k-1)}]
# Using 1/|h|^{2(k-1)} = (1/(k-2)!) ∫_0^∞ s^{k-2} e^{-s|h|²} ds and Wick's theorem:
#   ρ^(k) = U^{⊗k} D S U^{†⊗k},
#   D_{i1..ik} = (1/(k-2)!) ∫_0^∞ ds s^{k-2} Π_i (1+sλ_i)^{-1} Π_j λ_{ij}/(1+sλ_{ij})
# (valid for k ≥ 2; reduces to the Haar moment when ρ = 1/d).
# ----------------------------------------------------------------------------
function scrooge_weight(λ::Vector{Float64}, idxs::Vector{Int}, k::Int)
    f(s) = s^(k - 2) * prod(1 / (1 + s * l) for l in λ) *
           prod(λ[i] / (1 + s * λ[i]) for i in idxs)
    val, _ = quadgk(f, 0.0, Inf; rtol = 1e-10, atol = 1e-14)
    return val / factorial(k - 2)
end

function scrooge_moment(ρ::AbstractMatrix, k::Int, S::Matrix{Float64})
    @assert k ≥ 2
    d = size(ρ, 1)
    F = eigen(Hermitian(Matrix(ρ)))
    λ = clamp.(F.values, 0.0, Inf)
    λ ./= sum(λ)
    cache = Dict{Vector{Int},Float64}()
    Dvec = zeros(d^k)
    for idx in 0:d^k-1
        key = sort(tdigits(idx, d, k))
        Dvec[idx+1] = get!(() -> scrooge_weight(λ, key, k), cache, key)
    end
    Uk = kron_power(F.vectors, k)
    return Uk * (Dvec .* S) * Uk'
end

# ----------------------------------------------------------------------------
# Projected ensemble and measures
# ----------------------------------------------------------------------------
tracedist(A, B) = 0.5 * sum(abs, eigvals(Hermitian(A - B)))

"ρ^(k)_P from the unnormalized columns Ψ[:, z] = √p_z ψ_z."
function pe_moment(Ψ::AbstractMatrix, p::Vector{Float64}, k::Int)
    dA, dB = size(Ψ)
    keep = findall(>(1e-300), p)
    M = Matrix{ComplexF64}(undef, dA^k, length(keep))
    for (c, z) in enumerate(keep)
        v = Ψ[:, z] / sqrt(p[z])            # normalized projected state
        M[:, c] = kron_power(v, k) * sqrt(p[z])
    end
    return M * M'
end

"""
    measure_all(ψ, LA, L, cache; basis=:z)

Returns a NamedTuple with
  D1        = ½‖ρ_A − 1/d_A‖₁                 (first moment vs infinite temperature)
  Dk_haar   = ½‖ρ^(k)_P − ρ^(k)_Haar‖₁
  Dk_scr    = ½‖ρ^(k)_P − Scrooge[ρ_A]^(k)‖₁
  Dk_gs     = ½‖ρ^(k)_P − GS^(k)‖₁   (conditioned on # down spins in B; NaN unless basis=:z)
  Fk        = tr[(ρ^(k)_P)²] · binom(d_A+k−1, k)   (frame potential / Haar value)
"""
function measure_all(ψ::Vector{ComplexF64}, LA::Int, L::Int, cache; basis::Symbol = :z)
    dA = 2^LA; dB = 2^(L - LA)
    ψm = rotate_B(ψ, LA, L, basis)
    Ψ = reshape(ψm, dA, dB)
    p = vec(sum(abs2, Ψ; dims = 1))
    ρA = Ψ * Ψ'
    out = Dict{Symbol,Float64}()
    out[:D1] = tracedist(ρA, Matrix{ComplexF64}(I, dA, dA) / dA)

    # groups of outcomes by number of down spins in B (for the generalized Scrooge target)
    groups = Dict{Int,Vector{Int}}()
    for z in 1:dB
        push!(get!(groups, count_ones(z - 1), Int[]), z)
    end
    ρm = Dict{Int,Tuple{Float64,Matrix{ComplexF64}}}()
    if basis === :z
        for (m, zs) in groups
            pm = sum(p[zs])
            pm < 1e-14 && continue
            ρm[m] = (pm, (Ψ[:, zs] * Ψ[:, zs]') / pm)
        end
    end

    for k in cache.ks
        ρk = pe_moment(Ψ, p, k)
        out[Symbol("D$(k)_haar")] = tracedist(ρk, cache.haar[k])
        out[Symbol("D$(k)_scr")] = tracedist(ρk, scrooge_moment(ρA, k, cache.S[k]))
        if basis === :z
            gs = zeros(ComplexF64, dA^k, dA^k)
            for (_, (pm, r)) in ρm
                gs .+= pm .* scrooge_moment(r, k, cache.S[k])
            end
            out[Symbol("D$(k)_gs")] = tracedist(ρk, gs)
        else
            out[Symbol("D$(k)_gs")] = NaN
        end
        out[Symbol("F$(k)")] = real(tr(ρk * ρk)) * binomial(dA + k - 1, k)
    end
    return (; out...)
end

"""
    diagonal_ensemble_rhoA(E, V, ψ0, LA, L; tol=1e-9)

Reduced density matrix of the diagonal ensemble (exact infinite-time average of ρ_A),
ρ_DE = Σ_E P_E |ψ0⟩⟨ψ0| P_E, with degenerate levels grouped (|E-E'| < tol).
Needs the full spectrum, so only for small L (ED); this is the true finite-size
stationary first moment, which approaches 1/d_A only as L → ∞.
"""
function diagonal_ensemble_rhoA(E::Vector{Float64}, V::Matrix, ψ0::Vector{ComplexF64}, LA::Int, L::Int; tol = 1e-9)
    dA = 2^LA; dB = 2^(L - LA)
    c = V' * ψ0
    ρA = zeros(ComplexF64, dA, dA)
    n = length(E); a = 1
    while a ≤ n
        b = a
        while b < n && E[b+1] - E[a] < tol
            b += 1
        end
        φ = V[:, a:b] * c[a:b]
        Φ = reshape(φ, dA, dB)
        ρA .+= Φ * Φ'
        a = b + 1
    end
    return ρA
end

# ----------------------------------------------------------------------------
# Fixed late-time U(1) targets (z-basis measurement; m = # down spins measured in B)
# ----------------------------------------------------------------------------
"outcome indices z (1-based columns of Ψ) grouped by m = number of down spins in B"
function charge_groups(LA::Int, L::Int)
    LB = L - LA
    groups = [Int[] for _ in 0:LB]
    for z in 1:2^LB
        push!(groups[count_ones(z - 1)+1], z)
    end
    return groups
end

"""
    diagonal_ensemble_conditioned(E, V, ψ0, LA, L)

Charge-resolved stationary state: for each m,
    ρ̃_m = Tr_B[(1 ⊗ Π^B_m) ρ_DE],   p_m = tr ρ̃_m,   ρ_m = ρ̃_m / p_m,
with ρ_DE the diagonal ensemble (degenerate levels grouped). This is the exact
infinite-time average of the Q_B-resolved first moment. Returns Vector of (m, p_m, ρ_m).
"""
function diagonal_ensemble_conditioned(E::Vector{Float64}, V::Matrix, ψ0::Vector{ComplexF64},
                                       LA::Int, L::Int; tol = 1e-9)
    dA = 2^LA; dB = 2^(L - LA)
    groups = charge_groups(LA, L)
    acc = [zeros(ComplexF64, dA, dA) for _ in groups]
    c = V' * ψ0
    n = length(E); a = 1
    while a ≤ n
        b = a
        while b < n && E[b+1] - E[a] < tol
            b += 1
        end
        Φ = reshape(V[:, a:b] * c[a:b], dA, dB)
        for (g, zs) in enumerate(groups)
            Φg = Φ[:, zs]
            acc[g] .+= Φg * Φg'
        end
        a = b + 1
    end
    out = Tuple{Int,Float64,Matrix{ComplexF64}}[]
    for (g, ρ) in enumerate(acc)
        pm = real(tr(ρ))
        pm > 1e-14 && push!(out, (g - 1, pm, ρ / pm))
    end
    return out
end

"P(N): distribution of the total number of down spins in ψ (conserved by H)."
function charge_distribution(ψ::Vector{ComplexF64}, L::Int)
    P = zeros(L + 1)
    for s in 0:2^L-1
        P[count_ones(s)+1] += abs2(ψ[s+1])
    end
    return P
end

"""
    u1_conditioned(ψ0, LA, L)

U(1) maximum-entropy prediction: the global state is uniformly random inside each
charge sector N, with sector weights P(N) fixed by the initial state (the only
information U(1) conservation protects). Conditioned on m down spins in B,
    ρ_m ∝ Σ_{n_A} P(n_A+m) · binom(L_B,m)/binom(L,n_A+m) · Π^A_{n_A},
i.e. diagonal in the A charge sectors and maximally mixed inside each.
Returns Vector of (m, p_m, ρ_m).
"""
function u1_conditioned(ψ0::Vector{ComplexF64}, LA::Int, L::Int)
    LB = L - LA; dA = 2^LA
    P = charge_distribution(ψ0, L)
    out = Tuple{Int,Float64,Matrix{ComplexF64}}[]
    for m in 0:LB
        ρ = zeros(ComplexF64, dA, dA)
        for a in 0:dA-1
            N = count_ones(a) + m
            ρ[a+1, a+1] = P[N+1] * binomial(LB, m) / binomial(L, N)
        end
        pm = real(tr(ρ))
        pm > 1e-14 && push!(out, (m, pm, ρ / pm))
    end
    return out
end

"first moment Σ_m p_m ρ_m of a conditioned decomposition"
first_moment(cond) = sum(pm * ρ for (_, pm, ρ) in cond)

"generalized Scrooge k-th moment Σ_m p_m Scrooge[ρ_m]^(k)"
gs_moment(cond, k::Int, S::Matrix{Float64}) = sum(pm * scrooge_moment(ρ, k, S) for (_, pm, ρ) in cond)

"""
U(1)-random benchmark state: Haar-random inside each charge sector N, with the
sector weights P(N) of ψ0. Its projected ensemble realizes the U(1) target up to
finite-d_B effects, so it sets the floor for Δ^(k) against `u1_conditioned`.
"""
function u1_random_state(rng::AbstractRNG, ψ0::Vector{ComplexF64}, L::Int)
    P = charge_distribution(ψ0, L)
    ψ = zeros(ComplexF64, 2^L)
    for N in 0:L
        idx = [s + 1 for s in 0:2^L-1 if count_ones(s) == N]
        g = randn(rng, ComplexF64, length(idx))
        ψ[idx] = sqrt(P[N+1]) * g / norm(g)
    end
    return ψ
end

# ----------------------------------------------------------------------------
# Krylov time evolution with measurements at the requested times
# ----------------------------------------------------------------------------
"""
    evolve_measure(H, ψ0, times, f; tol=1e-12)

Evolves ψ(t) = e^{-iHt} ψ0 with a Krylov (Lanczos) exponential between consecutive
times and returns [f(ψ(t)) for t in times]. `times` must be sorted and ≥ 0.
"""
function evolve_measure(H, ψ0::Vector{ComplexF64}, times, f; tol = 1e-12)
    ψ = copy(ψ0)
    tprev = 0.0
    res = Any[]
    for t in times
        dt = t - tprev
        if dt > 0
            ψ, info = exponentiate(H, -im * dt, ψ; ishermitian = true, tol = tol)
            info.converged == 0 && @warn "Krylov exponentiate did not converge at t=$t"
            ψ ./= norm(ψ)
        end
        push!(res, f(ψ))
        tprev = t
    end
    return res
end

end # module
