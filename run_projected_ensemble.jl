# Projected-ensemble moments after a quench from random product states.
#
#   julia --project=. -t 2 run_projected_ensemble.jl [L] [LA] [nreal] [basis]
#
# Defaults: L=10, LA=2, nreal=100, basis=z.  Edit MODELS / TIMES below to change the scan.
# Writes results/raw_<tag>.csv (every realization and time) and results/summary_<tag>.csv
# (late-time averages with standard errors over realizations).

include("src/DeepTherm.jl")
using .DeepTherm, LinearAlgebra, Random, Printf, Statistics
BLAS.set_num_threads(1)                      # parallelize over realizations instead

L     = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 10
LA    = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 2
NREAL = length(ARGS) ≥ 3 ? parse(Int, ARGS[3]) : 100
BASIS = length(ARGS) ≥ 4 ? Symbol(ARGS[4]) : :z
KS    = 2:4
SEED  = 1234

# (name, Δ, J2).  Δ is the shift in (1+Δ) Z Z; Δ=-1 is XX (free fermions), Δ=0 is XXX.
MODELS = [
    ("XX_free",       -1.0, 0.0),
    ("XXZ_easyplane", -0.5, 0.0),
    ("XXX",            0.0, 0.0),
    ("XXZ_easyaxis",   0.5, 0.0),
    ("XXZ+NNN_chaotic", 0.5, 0.5),           # integrability broken, U(1) kept
]
TIMES  = vcat(0.0, 10 .^ range(-1, log10(200), length = 24))
T_LATE = 50.0                                 # late-time window t ≥ T_LATE

tag = "L$(L)_LA$(LA)_n$(NREAL)_$(BASIS)"
mkpath("results")
cache = moment_cache(2^LA, KS)
USE_DE = L ≤ 12          # diagonal ensemble needs full ED
keys_ = [[:D1, :D1_DE, :DE_inf]; vcat([[Symbol("D$(k)_haar"), Symbol("D$(k)_scr"), Symbol("D$(k)_gs"), Symbol("F$(k)")] for k in KS]...)]

rows = Vector{Vector{Any}}()                  # model, Δ, J2, realization, t, measures...
lk = ReentrantLock()

@printf("L=%d  LA=%d  realizations=%d  basis=%s  threads=%d\n", L, LA, NREAL, BASIS, Threads.nthreads())

for (name, Δ, J2) in MODELS
    H = build_hamiltonian(L; J = 1.0, Δ = Δ, J2 = J2)
    t0 = time()
    ED = USE_DE ? eigen(Hermitian(Matrix(H))) : nothing
    Threads.@threads for r in 1:NREAL
        rng = MersenneTwister(SEED + r)           # same initial states for every model
        ψ0 = random_product_state(rng, L)
        dA = 2^LA
        ρDE = USE_DE ? DeepTherm.diagonal_ensemble_rhoA(ED.values, ED.vectors, ψ0, LA, L) : nothing
        DEinf = USE_DE ? DeepTherm.tracedist(ρDE, Matrix{ComplexF64}(I, dA, dA) / dA) : NaN
        function f(ψ)
            m = measure_all(ψ, LA, L, cache; basis = BASIS)
            Ψ = reshape(ψ, dA, :)
            d1de = USE_DE ? DeepTherm.tracedist(Ψ * Ψ', ρDE) : NaN
            return merge(m, (D1_DE = d1de, DE_inf = DEinf))
        end
        res = DeepTherm.evolve_measure(H, ψ0, TIMES, f)
        lock(lk) do
            for (t, m) in zip(TIMES, res)
                push!(rows, Any[name, Δ, J2, r, t, (getfield(m, key) for key in keys_)...])
            end
        end
    end
    @printf("  %-16s done in %.1f s\n", name, time() - t0)
end

# Benchmark: Haar-random global states (no dynamics) — finite-L_B floor for every measure
Threads.@threads for r in 1:NREAL
    rng = MersenneTwister(10^6 + SEED + r)
    m = merge(measure_all(haar_state(rng, L), LA, L, cache; basis = BASIS), (D1_DE = NaN, DE_inf = NaN))
    lock(lk) do
        push!(rows, Any["Haar_global_state", NaN, NaN, r, Inf, (getfield(m, key) for key in keys_)...])
    end
end

# ---- write raw data
open("results/raw_$tag.csv", "w") do io
    println(io, join(["model", "Delta", "J2", "realization", "t", string.(keys_)...], ","))
    for row in sort(rows, by = x -> (x[1], x[4], x[5]))
        println(io, join(row, ","))
    end
end

# ---- late-time summary: time-average per realization over t ≥ T_LATE, then mean ± s.e.
open("results/summary_$tag.csv", "w") do io
    println(io, join(["model", "Delta", "J2", vcat([[string(k), string(k) * "_se"] for k in keys_]...)...], ","))
    println("\nLate-time averages (t ≥ $T_LATE), mean ± s.e. over $NREAL realizations:")
    @printf("%-18s", "model"); foreach(k -> @printf("%16s", k), keys_); println()
    for name in vcat(first.(MODELS), "Haar_global_state")
        sel = filter(x -> x[1] == name && x[5] ≥ T_LATE, rows)
        Δ, J2 = sel[1][2], sel[1][3]
        line = Any[name, Δ, J2]
        @printf("%-18s", name)
        for (j, key) in enumerate(keys_)
            per_real = [mean(x[5+j] for x in sel if x[4] == r) for r in 1:NREAL]
            μ, se = mean(per_real), std(per_real) / sqrt(NREAL)
            push!(line, μ, se)
            @printf("%16s", isnan(μ) ? "—" : @sprintf("%.4f±%.4f", μ, se))
        end
        println(); println(io, join(line, ","))
    end
end
println("\nwrote results/raw_$tag.csv and results/summary_$tag.csv")
