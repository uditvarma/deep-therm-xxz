# Projected-ensemble moments against FIXED late-time U(1) targets.
#
#   julia --project=. -t 2 run_u1_scrooge.jl [L] [LA] [nreal] [Δ] [J2]
#
# Defaults: L=10, LA=2, nreal=100, Δ=0.5 (i.e. 1+Δ = 1.5), J2=0. z-basis measurement of B.
#
# Targets (fixed in time, one per realization), m = number of down spins measured in B:
#   GS_DE : Σ_m p_m Scrooge[ρ_m^DE],  ρ_m^DE = charge-resolved diagonal ensemble
#           (exact infinite-time average of the Q_B-resolved first moment; contains all
#            conserved-quantity information, incl. energy and the integrable charges' effect
#            on the first moment)
#   GS_U1 : Σ_m p_m Scrooge[ρ_m^U1],  ρ_m^U1 = U(1) maximum-entropy state fixed only by the
#           conserved charge distribution P(N) of the initial state
# Floor: U(1)-random states (Haar inside each charge sector, same P(N)).
#
# Writes results/u1_raw_<tag>.csv and results/u1_summary_<tag>.csv.

include("src/DeepTherm.jl")
using .DeepTherm, LinearAlgebra, Random, Printf, Statistics
const DT = DeepTherm
BLAS.set_num_threads(1)

L     = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 10
LA    = length(ARGS) ≥ 2 ? parse(Int, ARGS[2]) : 2
NREAL = length(ARGS) ≥ 3 ? parse(Int, ARGS[3]) : 100
Δ     = length(ARGS) ≥ 4 ? parse(Float64, ARGS[4]) : 0.5
J2    = length(ARGS) ≥ 5 ? parse(Float64, ARGS[5]) : 0.0
KS    = 2:4
SEED  = 1234
TIMES = vcat(0.0, 10 .^ range(-1, log10(200), length = 30))
T_LATE = 50.0
@assert L ≤ 12 "the diagonal-ensemble target needs full ED; use L ≤ 12 (or ≤ 14 with patience)"

dA = 2^LA
tag = "L$(L)_LA$(LA)_n$(NREAL)_D$(Δ)_J2$(J2)"
mkpath("results")
cache = moment_cache(dA, KS)
Id = Matrix{ComplexF64}(I, dA, dA) / dA

keys_ = [[:D1_inf, :D1_DE, :D1_U1];
         vcat([[Symbol("D$(k)_haar"), Symbol("D$(k)_gsDE"), Symbol("D$(k)_gsU1"), Symbol("F$(k)")] for k in KS]...);
         [:T1_DEvsU1]; [Symbol("T$(k)_DEvsU1") for k in KS];
         [Symbol("FgsDE$(k)") for k in KS]; [Symbol("FgsU1$(k)") for k in KS]]

"all measures of ψ against the fixed targets `tg`"
function measures(ψ, tg)
    Ψ = reshape(ψ, dA, :)
    p = vec(sum(abs2, Ψ; dims = 1))
    ρA = Ψ * Ψ'
    out = Dict{Symbol,Float64}(
        :D1_inf => DT.tracedist(ρA, Id),
        :D1_DE  => tg.ρA_DE === nothing ? NaN : DT.tracedist(ρA, tg.ρA_DE),
        :D1_U1  => DT.tracedist(ρA, tg.ρA_U1))
    for k in KS
        ρk = DT.pe_moment(Ψ, p, k)
        out[Symbol("D$(k)_haar")] = DT.tracedist(ρk, cache.haar[k])
        out[Symbol("D$(k)_gsDE")] = tg.gsDE === nothing ? NaN : DT.tracedist(ρk, tg.gsDE[k])
        out[Symbol("D$(k)_gsU1")] = DT.tracedist(ρk, tg.gsU1[k])
        out[Symbol("F$(k)")] = real(tr(ρk * ρk)) * binomial(dA + k - 1, k)
    end
    merge!(out, tg.consts)
    return out
end

"fixed targets for one realization (condDE = nothing for the floor benchmark)"
function targets(condDE, condU1)
    gsU1 = Dict(k => DT.gs_moment(condU1, k, cache.S[k]) for k in KS)
    ρA_U1 = DT.first_moment(condU1)
    consts = Dict{Symbol,Float64}()
    for k in KS
        consts[Symbol("FgsU1$(k)")] = real(tr(gsU1[k]^2)) * binomial(dA + k - 1, k)
    end
    if condDE === nothing
        gsDE = nothing; ρA_DE = nothing
        consts[:T1_DEvsU1] = NaN
        for k in KS
            consts[Symbol("T$(k)_DEvsU1")] = NaN; consts[Symbol("FgsDE$(k)")] = NaN
        end
    else
        gsDE = Dict(k => DT.gs_moment(condDE, k, cache.S[k]) for k in KS)
        ρA_DE = DT.first_moment(condDE)
        consts[:T1_DEvsU1] = DT.tracedist(ρA_DE, ρA_U1)
        for k in KS
            consts[Symbol("T$(k)_DEvsU1")] = DT.tracedist(gsDE[k], gsU1[k])
            consts[Symbol("FgsDE$(k)")] = real(tr(gsDE[k]^2)) * binomial(dA + k - 1, k)
        end
    end
    return (; gsDE, gsU1, ρA_DE, ρA_U1, consts)
end

@printf("XXZ (1+Δ)=%.3g, J2=%.3g | L=%d LA=%d realizations=%d threads=%d\n",
        1 + Δ, J2, L, LA, NREAL, Threads.nthreads())
H = build_hamiltonian(L; J = 1.0, Δ = Δ, J2 = J2)
ED = eigen(Hermitian(Matrix(H)))

rows = Vector{Vector{Any}}()
lk = ReentrantLock()
t0 = time()
Threads.@threads for r in 1:NREAL
    rng = MersenneTwister(SEED + r)
    ψ0 = random_product_state(rng, L)
    condDE = DT.diagonal_ensemble_conditioned(ED.values, ED.vectors, ψ0, LA, L)
    condU1 = DT.u1_conditioned(ψ0, LA, L)
    # consistency: Σ_m p_m ρ_m^DE must equal the unconditioned diagonal-ensemble ρ_A
    @assert norm(DT.first_moment(condDE) -
                 DT.diagonal_ensemble_rhoA(ED.values, ED.vectors, ψ0, LA, L)) < 1e-10
    tg = targets(condDE, condU1)
    res = DT.evolve_measure(H, ψ0, TIMES, ψ -> measures(ψ, tg))
    # floor: U(1)-random state with the same P(N), measured against its U(1) target
    ψU = DT.u1_random_state(MersenneTwister(10^6 + SEED + r), ψ0, L)
    mU = measures(ψU, targets(nothing, condU1))
    lock(lk) do
        for (t, m) in zip(TIMES, res)
            push!(rows, Any["XXZ", r, t, (m[key] for key in keys_)...])
        end
        push!(rows, Any["U1_random_state", r, Inf, (mU[key] for key in keys_)...])
    end
end
@printf("done in %.1f s\n", time() - t0)

open("results/u1_raw_$tag.csv", "w") do io
    println(io, join(["model", "realization", "t", string.(keys_)...], ","))
    for row in sort(rows, by = x -> (x[1], x[2], x[3]))
        println(io, join(row, ","))
    end
end

open("results/u1_summary_$tag.csv", "w") do io
    println(io, "quantity,XXZ_late_mean,XXZ_late_se,U1_floor_mean,U1_floor_se")
    @printf("\nLate-time (t ≥ %g) mean ± s.e. over %d realizations   |   U(1)-random-state floor\n", T_LATE, NREAL)
    for (j, key) in enumerate(keys_)
        late = [mean(x[3+j] for x in rows if x[1] == "XXZ" && x[2] == r && x[3] ≥ T_LATE) for r in 1:NREAL]
        flo = [x[3+j] for x in rows if x[1] == "U1_random_state"]
        μ, se = mean(late), std(late) / sqrt(NREAL)
        μf, sef = mean(flo), std(flo) / sqrt(NREAL)
        println(io, join([key, μ, se, μf, sef], ","))
        @printf("  %-12s %8.4f ± %.4f   |   %s\n", key, μ, se,
                isnan(μf) ? "—" : @sprintf("%.4f ± %.4f", μf, sef))
    end
end
println("\nwrote results/u1_raw_$tag.csv and results/u1_summary_$tag.csv")
