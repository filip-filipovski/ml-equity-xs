"""Forward-evidence power for the ranking-signal claim (closing question 7: what would you want before sizing it).
Inputs: p9_b_followups.json — monthly C1 rank IC 2009–2023 (the 168 months 2009–22 are the prior window) and the
2009–2023 decile L/S net monthly series (15 bps). No new return read: every number is a function of series already in
the kit. Question answered: how many FORWARD months (post-sealed / live paper) settle (a) "IC > 0", (b) "IC at the
2019–23 level (0.06) rather than the 2009–22 level (0.03)", (c) "book Sharpe > 0" at a stated Sharpe, each at one-sided
5 % size and 80 / 90 % power. Analytic normal approximation + block-3 bootstrap from the demeaned real series (the same
null machinery as the pass-probability study). Writes p9_b_forward_power.json."""
import json, math, sys
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
f = json.load(open(HERE / "p9_b_followups.json"))
icm = {int(k): v for k, v in f["ic_monthly_2009_2023"].items()}
ret = {int(k): v for k, v in f["decile_net_monthly"].items()} if isinstance(f.get("decile_net_monthly"), dict) else None
# tmi = months since 1970 (2009-01 = 24108). 2009-01..2022-12 = 24108..24275
ic22 = np.array([icm[t] for t in sorted(icm) if t <= 24275]); assert len(ic22) == 168, len(ic22)
mu0, sd0, rho = ic22.mean(), ic22.std(ddof=1), np.corrcoef(ic22[:-1], ic22[1:])[0, 1]
log = lambda *a: print(*a, flush=True)
log(f"IC 2009-22: n {len(ic22)} mean {mu0:+.4f} sd {sd0:.4f} lag1 {rho:+.3f}")

za = stats.norm.ppf(0.95)
def n_analytic(delta, sd, power):
    zb = stats.norm.ppf(power); return int(math.ceil(((za + zb) * sd / delta) ** 2))
def power_at(delta, sd, n):
    return float(1 - stats.norm.cdf(za - delta * math.sqrt(n) / sd))

rng = np.random.default_rng(20261013)
def block_boot_power(base, delta, n, B=4000, L=3):
    """Power of the one-sided t-test (H0 mean <= 0, 5 %) on n months drawn by circular block-L bootstrap from the
    demeaned real series, with mean delta planted."""
    z = base - base.mean(); N = len(z); nb = math.ceil(n / L)
    starts = rng.integers(0, N, size=(B, nb)); idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :n] % N
    x = z[idx] + delta; m = x.mean(1); s = x.std(1, ddof=1); t = m / (s / math.sqrt(n))
    return float((t >= stats.t.ppf(0.95, n - 1)).mean())

out = {"design": __doc__, "ic_2009_2022": {"n": 168, "mean": mu0, "sd": sd0, "lag1": rho, "test": "one-sided 5 %, H0: mean IC <= reference"},
       "ic_months_needed": [], "ic_power_at_n": [], "sharpe_months_needed": [], "sharpe_power_at_n": []}
log("\n(a)/(b) forward months needed, IC test (analytic / block-3 bootstrap from the real demeaned series)")
log(f"{'claim':<34}{'n80 analytic':>14}{'n80 boot':>10}{'n90 analytic':>14}")
grid = [3, 6, 9, 11, 12, 18, 24, 36, 48, 60, 72, 84, 96, 120, 180, 240]
for label, delta in [("IC > 0 at true 0.02", 0.02), ("IC > 0 at true 0.03 (2009-22 level)", 0.03), ("IC > 0 at true 0.036", 0.036),
                     ("IC > 0 at true 0.06 (2019-23 level)", 0.06), ("IC > 0 at true 0.087 (2023)", 0.087),
                     ("IC > 0.03 at true 0.06 (level change)", 0.03)]:
    n80, n90 = n_analytic(delta, sd0, 0.8), n_analytic(delta, sd0, 0.9)
    boot = {n: block_boot_power(ic22, delta, n) for n in grid}
    n80b = next((n for n in grid if boot[n] >= 0.8), None)
    out["ic_months_needed"].append({"claim": label, "delta": delta, "n80_analytic": n80, "n90_analytic": n90, "n80_boot_grid": n80b, "boot_power_by_n": boot})
    log(f"{label:<34}{n80:>14}{str(n80b):>10}{n90:>14}")
log("\npower at fixed horizons (block bootstrap): months 6 (author token 2024-H1) / 11 (the 2023 read) / 12 / 24 / 36 / 60")
for row in out["ic_months_needed"]:
    b = row["boot_power_by_n"]; log(f"{row['claim']:<34}" + "".join(f"{100*b[n]:>7.0f}%" for n in (6, 11, 12, 24, 36, 60)))
    out["ic_power_at_n"].append({"claim": row["claim"], **{f"m{n}": b[n] for n in (6, 11, 12, 24, 36, 60)}})

log("\n(c) forward months needed to show book Sharpe > 0 (one-sided 5 %), monthly data")
sr_grid = [0.2, 0.3, 0.5, 1.0, 1.5, 2.0]
if ret is not None:
    r22 = np.array([ret[t] for t in sorted(ret) if t <= 24275]); sdm = r22.std(ddof=1)
    log(f"decile net 2009-22: n {len(r22)} mean {r22.mean()*100:+.2f} %/mo sd {sdm*100:.2f} %/mo (SR {r22.mean()/sdm*math.sqrt(12):.2f})")
for sr in sr_grid:
    srm = sr / math.sqrt(12); n80, n90 = n_analytic(srm, 1.0, 0.8), n_analytic(srm, 1.0, 0.9)
    row = {"sharpe": sr, "n80_analytic_months": n80, "n80_years": round(n80 / 12, 1), "n90_analytic_months": n90, "n90_years": round(n90 / 12, 1)}
    if ret is not None:
        boot = {n: block_boot_power(r22, srm * sdm, n) for n in grid}; row["boot_power_by_n"] = boot
        row["n80_boot_grid"] = next((n for n in grid if boot[n] >= 0.8), None)
        out["sharpe_power_at_n"].append({"sharpe": sr, **{f"m{n}": boot[n] for n in (6, 11, 12, 24, 36, 60, 120)}})
    out["sharpe_months_needed"].append(row)
    log(f"SR {sr:<4} n80 {n80:>5} mo ({n80/12:5.1f} y)  n90 {n90:>5} mo ({n90/12:5.1f} y)  boot n80 {row.get('n80_boot_grid')}  power@12/36/60 mo " +
        (" / ".join(f"{100*row['boot_power_by_n'][n]:.0f}%" for n in (12, 36, 60)) if ret is not None else "-"))
(HERE / "p9_b_forward_power.json").write_text(json.dumps(out, indent=1, default=float)); log("wrote p9_b_forward_power.json")
