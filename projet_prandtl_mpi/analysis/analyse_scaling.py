#!/usr/bin/env python3
"""Analyse de passage a l'echelle a partir des CSV produits par bin/prandtl.

  python3 analysis/analyse_scaling.py data/strong_N8192.csv --mode strong
  python3 analysis/analyse_scaling.py data/weak_N0_4096.csv --mode weak

Le temps retenu est le temps PAR ITERATION (t_min / iterations) : le nombre d'iterations du
gradient conjugue croit avec N (~ sqrt(N)), ce qui fausserait le passage a l'echelle faible.
  fort  : S_P = T1/TP, E_P = S_P/P, fraction sequentielle f (Amdahl, moindres carres),
          metrique de Karp-Flatt e_P = (1/S - 1/P)/(1 - 1/P).
  faible: E_P = T1/TP, speedup etendu S_P = P T1/TP, f de Gustafson : S = P - f (P - 1).
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker


def fit_amdahl(P, S):
    """1/S - 1/P = f (1 - 1/P) : regression lineaire sans terme constant."""
    m = P > 1
    x, y = 1 - 1/P[m], 1/S[m] - 1/P[m]
    return float(np.clip(np.sum(x*y)/np.sum(x*x), 0.0, 1.0))


def analyser(df, mode):
    df = df.sort_values("P").copy()
    t = (df["t_min"]/df["iterations"]).to_numpy()
    P = df["P"].to_numpy(float)
    t1 = t[P == 1][0]
    out = pd.DataFrame({"P": df["P"].to_numpy(), "N": df["N"].to_numpy(), "t_iter_ms": 1e3*t})
    if mode == "strong":
        S = t1/t
        out["speedup"], out["efficacite"] = S, S/P
        with np.errstate(divide="ignore", invalid="ignore"):
            out["karp_flatt"] = np.where(P > 1, (1/S - 1/P)/(1 - 1/P), np.nan)
        out.attrs["f"] = fit_amdahl(P, S)
    else:
        out["efficacite"] = t1/t
        out["speedup_etendu"] = P*t1/t
        with np.errstate(divide="ignore", invalid="ignore"):
            out["f_gustafson"] = np.where(P > 1, (P - P*t1/t)/(P - 1), np.nan)
        out.attrs["f"] = float(np.nanmean(out["f_gustafson"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--mode", choices=["strong", "weak"], required=True)
    a = ap.parse_args()
    data = pd.read_csv(a.csv, skipinitialspace=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for ov, label in ((0, "bloquant (Allgatherv)"), (1, "recouvrement (Iallgatherv)")):
        sub = data[data["recouvrement"] == ov]
        if sub.empty or 1 not in set(sub["P"]):
            continue
        r = analyser(sub, a.mode)
        f = r.attrs["f"]
        print(f"\n=== {a.mode} scaling, {label} ===")
        print(r.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        if a.mode == "strong":
            print(f"fraction sequentielle (Amdahl) f = {f:.4f}  ->  speedup asymptotique 1/f = {1/f:.1f}")
            ax[0].plot(r["P"], r["speedup"], "o-", label=f"{label}, f={f:.3f}")
        else:
            print(f"fraction sequentielle moyenne (Gustafson) f = {f:.4f}")
            ax[0].plot(r["P"], r["speedup_etendu"], "o-", label=f"{label}, f={f:.3f}")
        ax[1].plot(r["P"], r["efficacite"], "o-", label=label)
    Pmax = data["P"].max()
    ax[0].plot([1, Pmax], [1, Pmax], "k--", label="ideal")
    ax[0].set(xlabel="P", ylabel="speedup" if a.mode == "strong" else "speedup etendu",
              xscale="log", yscale="log", title="Passage a l'echelle " + ("fort" if a.mode == "strong" else "faible"))
    ax[1].axhline(0.6, color="r", ls=":", label="seuil 60 %")
    ax[1].set(xlabel="P", ylabel="efficacite", xscale="log", ylim=(0, 1.1), title="Efficacite parallele")
    procs = sorted(set(data["P"]))
    for x in ax:
        x.set_xticks(procs)
        x.set_xticklabels([str(int(q)) for q in procs])
        x.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        x.grid(alpha=0.3, which="both")
        x.legend(fontsize=8)
    ax[0].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[0].yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    fig.tight_layout()
    png = a.csv.rsplit(".", 1)[0] + "_analyse.png"
    fig.savefig(png, dpi=130)
    print(f"\nFigure : {png}")


if __name__ == "__main__":
    main()
