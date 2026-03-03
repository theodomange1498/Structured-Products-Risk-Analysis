"""
BLOC 2 — Payoff Athena sur Heston calibré
==========================================
Ce fichier branche la logique Athena sur les trajectoires issues
du Bloc 1 calibré sur données réelles.

Deux modes :
  MODE A — Indice unique  (ex: EuroStoxx 50, CAC 40, S&P 500)
  MODE B — Panier personnalisé (ex: LVMH + Total + Airbus)

Dans les deux cas, les paramètres Heston sont calibrés automatiquement
sur les données historiques du sous-jacent choisi. On ne simule plus
des "régimes génériques" — on simule la vraie dynamique de volatilité
du sous-jacent.

Ce qu'on calcule :
  1. Payoff exact pour chaque trajectoire
  2. P(barrière protection touchée) — la métrique clé client
  3. Distribution des durées de vie du produit
  4. Sensibilité de P(barrière) au niveau de barrière choisi
  5. Comparaison indice vs panier
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
import sys
import os

# ── Import du Bloc 1 calibré ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heston_simulator import (
    get_index, get_basket,
    compute_realized_variance, calibrate_heston,
    simulate_heston, print_calibration
)


# ─────────────────────────────────────────────
# PARAMÈTRES DU PRODUIT ATHENA
# ─────────────────────────────────────────────
ATHENA_DEFAULT = {
    "S0":               100.0,
    "T":                8,
    "coupon_annuel":    0.06,
    "barriere_rappel":  1.00,
    "barriere_protect": 0.60,
    "nominal":          100.0,
}


# ─────────────────────────────────────────────
# MOTEUR DE PAYOFF ATHENA
# ─────────────────────────────────────────────
def pricer_athena(S, params=None, steps_per_year=52):
    """
    Calcule le payoff de l'Athena pour chaque trajectoire simulée.

    Paramètres :
        S              : array (M, N+1) — trajectoires de prix (base 100)
        params         : dict de paramètres produit (voir ATHENA_DEFAULT)
        steps_per_year : doit correspondre à simulate_heston

    Retourne un dict avec toutes les métriques.
    """
    if params is None:
        params = ATHENA_DEFAULT

    M          = S.shape[0]
    T          = params["T"]
    S0         = params["S0"]
    coupon     = params["coupon_annuel"]
    b_rappel   = params["barriere_rappel"]
    b_protect  = params["barriere_protect"]
    nominal    = params["nominal"]

    niveau_rappel  = S0 * b_rappel
    niveau_protect = S0 * b_protect

    constat_idx = [k * steps_per_year for k in range(1, T + 1)]

    payoffs      = np.zeros(M)
    recall_year  = np.zeros(M, dtype=int)
    barrier_time = np.full(M, np.nan)

    # ── Barrière de protection (américaine) ──
    below_barrier   = S < niveau_protect
    ever_breached   = below_barrier.any(axis=1)
    first_breach    = np.argmax(below_barrier, axis=1)
    barrier_touched = ever_breached
    barrier_time[ever_breached] = first_breach[ever_breached] / steps_per_year

    # ── Logique de rappel annuel ──
    recalled = np.zeros(M, dtype=bool)
    for k, idx in enumerate(constat_idx, start=1):
        S_k  = S[:, idx]
        cond = (~recalled) & (S_k >= niveau_rappel)
        payoffs[cond]     = nominal * (1 + coupon * k)
        recall_year[cond] = k
        recalled[cond]    = True

    # ── Payoff à maturité pour les non rappelés ──
    not_recalled = ~recalled
    S_final      = S[:, -1]
    safe_at_mat  = not_recalled & (S_final >= niveau_protect)
    loss_at_mat  = not_recalled & (S_final < niveau_protect)
    payoffs[safe_at_mat] = nominal
    payoffs[loss_at_mat] = nominal * (S_final[loss_at_mat] / S0)

    recall_proba = np.array([np.mean(recall_year == k) for k in range(1, T + 1)])

    details = {
        "P_rappel":          float(np.mean(recalled)),
        "P_barriere":        float(np.mean(barrier_touched)),
        "P_perte_maturite":  float(np.mean(loss_at_mat)),
        "payoff_moyen":      float(np.mean(payoffs)),
        "payoff_median":     float(np.median(payoffs)),
        "payoff_p10":        float(np.percentile(payoffs, 10)),
        "payoff_p90":        float(np.percentile(payoffs, 90)),
        "duree_vie_moyenne": float(np.mean(recall_year[recalled])) if recalled.any() else float(T),
        "recall_proba":      recall_proba,
        "taux_actualise":    _taux_annualise(payoffs, recall_year, recalled, T, nominal),
    }

    return {
        "payoffs":         payoffs,
        "recall_year":     recall_year,
        "barrier_touched": barrier_touched,
        "barrier_time":    barrier_time,
        "recall_proba":    recall_proba,
        "details":         details,
    }


def _taux_annualise(payoffs, recall_year, recalled, T, nominal):
    durees = np.where(recalled, recall_year, T).astype(float)
    durees = np.maximum(durees, 0.001)
    taux   = (payoffs / nominal) ** (1.0 / durees) - 1.0
    return float(np.mean(taux) * 100)


# ─────────────────────────────────────────────
# AFFICHAGE CONSOLE
# ─────────────────────────────────────────────
def print_stats_athena(results, label, heston_params, params=None):
    if params is None:
        params = ATHENA_DEFAULT
    d   = results["details"]
    b   = int(params["barriere_protect"] * 100)
    c   = int(params["coupon_annuel"] * 100)
    T   = params["T"]
    v0  = np.sqrt(heston_params["v0"]) * 100
    th  = np.sqrt(heston_params["theta"]) * 100

    print(f"\n{'='*65}")
    print(f"  ATHENA — {label}")
    print(f"  Heston : vol initiale={v0:.1f}%  vol LT={th:.1f}%  "
          f"xi={heston_params['xi']}  rho={heston_params['rho']}")
    print(f"  Produit : barriere {b}%  |  coupon {c}%/an  |  maturite {T} ans")
    print(f"{'='*65}")
    print(f"  P(rappel anticipe)             : {d['P_rappel']*100:.1f}%")
    print(f"  Duree de vie moyenne           : {d['duree_vie_moyenne']:.1f} ans")
    print(f"  P(barriere protection touchee) : {d['P_barriere']*100:.1f}%  <- METRIQUE CLE")
    print(f"  P(perte a maturite)            : {d['P_perte_maturite']*100:.1f}%")
    print(f"  Payoff moyen                   : {d['payoff_moyen']:.1f}  (investi: 100)")
    print(f"  Payoff P10 / P90               : {d['payoff_p10']:.1f} / {d['payoff_p90']:.1f}")
    print(f"  Taux annualise equivalent moy. : {d['taux_actualise']:.2f}%  (vs 3% sans risque)")
    print(f"\n  Probabilites de rappel par annee :")
    for k, p in enumerate(d["recall_proba"], 1):
        bar = "X" * int(p * 40)
        print(f"    Annee {k} : {p*100:5.1f}%  {bar}")
    print(f"{'='*65}")


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────
def plot_athena_single(S, results, heston_params, label, params=None):
    """
    4 graphiques pour un sous-jacent :
    - Haut gauche  : Distribution des payoffs (KDE)
    - Haut droite  : Sensibilite P(barriere) vs niveau de barriere
    - Bas gauche   : Probabilites de rappel par annee
    - Bas droite   : P(barriere touchee) cumulee dans le temps
    """
    if params is None:
        params = ATHENA_DEFAULT

    T              = params["T"]
    steps          = S.shape[1] - 1
    steps_per_year = steps // T
    niveau_protect = 100 * params["barriere_protect"]
    d              = results["details"]

    fig  = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor("#0f1117")
    gs   = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
    ax1, ax2, ax3, ax4 = axes

    COLOR = "#4a9eff"

    # ── 1. Distribution des payoffs (KDE) ──
    payoffs = results["payoffs"]
    kde     = gaussian_kde(payoffs, bw_method=0.15)
    x_range = np.linspace(payoffs.min(), payoffs.max(), 500)
    ax1.plot(x_range, kde(x_range), color=COLOR, linewidth=2.5)
    ax1.fill_between(x_range, kde(x_range), alpha=0.2, color=COLOR)
    ax1.axvline(100, color="white",    linestyle="--", linewidth=1.2, label="Capital initial")
    ax1.axvline(niveau_protect, color="#ff6b6b", linestyle="--",
                linewidth=1.2, label=f"Barriere {int(niveau_protect)}%")
    ax1.axvline(d["payoff_p10"], color="#ffcc00", linestyle=":",
                linewidth=1.2, label=f"P10 = {d['payoff_p10']:.0f}")
    ax1.set_title("Distribution des payoffs", fontsize=11)
    ax1.set_xlabel("Payoff final (investi = 100)")
    ax1.set_ylabel("Densite")
    ax1.legend(fontsize=8, facecolor="#1a1d27", labelcolor="white")

    # ── 2. Sensibilite P(barriere) vs niveau ──
    niveaux   = np.arange(0.40, 0.86, 0.05)
    p_barrier = [np.mean((S < 100 * niv).any(axis=1)) * 100 for niv in niveaux]
    ax2.plot(niveaux * 100, p_barrier, color=COLOR, linewidth=2.5, marker="o", markersize=5)
    ax2.fill_between(niveaux * 100, p_barrier, alpha=0.15, color=COLOR)
    ax2.axvline(params["barriere_protect"] * 100, color="#ff6b6b",
                linestyle="--", linewidth=1.5,
                label=f"Barriere actuelle ({int(params['barriere_protect']*100)}%) "
                      f"-> {d['P_barriere']*100:.1f}%")
    ax2.set_title("Sensibilite : P(barriere touchee) vs niveau barriere", fontsize=11)
    ax2.set_xlabel("Niveau de barriere (% du niveau initial)")
    ax2.set_ylabel("P(barriere touchee) (%)")
    ax2.legend(fontsize=8, facecolor="#1a1d27", labelcolor="white")
    ax2.grid(True, alpha=0.15)

    # ── 3. Probabilites de rappel par annee ──
    rp   = d["recall_proba"] * 100
    x    = np.arange(1, T + 1)
    bars = ax3.bar(x, rp, color=COLOR, alpha=0.85, width=0.6)
    for bar, val in zip(bars, rp):
        if val > 0.5:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val:.1f}%", ha="center", va="bottom",
                     color="white", fontsize=8)
    ax3.set_title("Probabilite de rappel par annee", fontsize=11)
    ax3.set_xlabel("Annee de constatation")
    ax3.set_ylabel("Probabilite (%)")
    ax3.set_xticks(x)
    ax3.grid(True, alpha=0.15, axis="y")

    # ── 4. P(barriere touchee) cumulee dans le temps ──
    pas = max(steps_per_year // 4, 1)
    cumul = np.array([
        np.mean((S[:, :i+1] < niveau_protect).any(axis=1))
        for i in range(0, S.shape[1], pas)
    ]) * 100
    t_trim = np.linspace(0, T, len(cumul))
    ax4.plot(t_trim, cumul, color=COLOR, linewidth=2.5)
    ax4.fill_between(t_trim, cumul, alpha=0.15, color=COLOR)
    ax4.scatter([t_trim[-1]], [cumul[-1]], color="#ff6b6b", zorder=5,
                label=f"Final : {cumul[-1]:.1f}%")
    ax4.set_title("P(barriere touchee) cumulee dans le temps", fontsize=11)
    ax4.set_xlabel("Annees depuis emission")
    ax4.set_ylabel("P(barriere touchee au moins une fois) (%)")
    ax4.legend(fontsize=8, facecolor="#1a1d27", labelcolor="white")
    ax4.grid(True, alpha=0.15)

    v0_pct = np.sqrt(heston_params["v0"]) * 100
    th_pct = np.sqrt(heston_params["theta"]) * 100
    b      = int(params["barriere_protect"] * 100)
    c      = int(params["coupon_annuel"] * 100)
    fig.suptitle(
        f"Analyse Athena — {label}\n"
        f"Heston calibre : vol0={v0_pct:.1f}%  theta={th_pct:.1f}%  "
        f"xi={heston_params['xi']}  rho={heston_params['rho']}  "
        f"|  Barriere {b}%  Coupon {c}%/an  Maturite {T} ans  "
        f"|  {S.shape[0]:,} simulations",
        color="white", fontsize=10, y=1.01
    )

    out = f"athena_{label.replace(' ', '_').replace('/', '_')}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Graphique sauvegarde : {out}")


def plot_comparison(results_dict, S_dict, params=None):
    """
    Comparaison visuelle entre deux sous-jacents sur 3 panels :
    - KDE des payoffs superposes
    - Metriques cles cote a cote
    - Sensibilite P(barriere) pour les deux
    """
    if params is None:
        params = ATHENA_DEFAULT

    labels = list(results_dict.keys())
    colors = ["#4a9eff", "#ff6b6b", "#4ec9b0", "#ffcc00"]
    T      = params["T"]

    fig  = plt.figure(figsize=(16, 6))
    fig.patch.set_facecolor("#0f1117")
    gs   = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)
    axes = [fig.add_subplot(gs[0, j]) for j in range(3)]
    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
    ax1, ax2, ax3 = axes

    # ── KDE superposes ──
    for i, lbl in enumerate(labels):
        payoffs = results_dict[lbl]["payoffs"]
        kde     = gaussian_kde(payoffs, bw_method=0.15)
        x_range = np.linspace(payoffs.min(), payoffs.max(), 500)
        ax1.plot(x_range, kde(x_range), color=colors[i], linewidth=2.5, label=lbl)
        ax1.fill_between(x_range, kde(x_range), alpha=0.12, color=colors[i])
    ax1.axvline(100, color="white", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_title("Distribution des payoffs", fontsize=11)
    ax1.set_xlabel("Payoff final")
    ax1.legend(fontsize=8, facecolor="#1a1d27", labelcolor="white")

    # ── Metriques cles ──
    metriques = ["P_barriere", "P_rappel", "P_perte_maturite"]
    labels_m  = ["P(barriere\ntouchee)", "P(rappel\nanticipe)", "P(perte\na maturite)"]
    x_pos     = np.arange(len(metriques))
    width     = 0.35
    for i, lbl in enumerate(labels):
        d       = results_dict[lbl]["details"]
        valeurs = [d[m] * 100 for m in metriques]
        offset  = (i - len(labels)/2 + 0.5) * width
        bars    = ax2.bar(x_pos + offset, valeurs, width=width*0.9,
                          color=colors[i], alpha=0.85, label=lbl)
        for bar, val in zip(bars, valeurs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val:.1f}%", ha="center", va="bottom",
                     color="white", fontsize=7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels_m, color="white", fontsize=9)
    ax2.set_title("Comparaison metriques cles", fontsize=11)
    ax2.set_ylabel("Probabilite (%)")
    ax2.legend(fontsize=8, facecolor="#1a1d27", labelcolor="white")
    ax2.grid(True, alpha=0.15, axis="y")

    # ── Sensibilite P(barriere) ──
    niveaux = np.arange(0.40, 0.86, 0.05)
    for i, lbl in enumerate(labels):
        S         = S_dict[lbl]
        p_barrier = [np.mean((S < 100 * niv).any(axis=1)) * 100 for niv in niveaux]
        ax3.plot(niveaux * 100, p_barrier, color=colors[i],
                 linewidth=2.5, marker="o", markersize=4, label=lbl)
    ax3.axvline(params["barriere_protect"] * 100, color="white",
                linestyle="--", linewidth=1, alpha=0.7,
                label=f"Barriere actuelle ({int(params['barriere_protect']*100)}%)")
    ax3.set_title("Sensibilite P(barriere) vs niveau", fontsize=11)
    ax3.set_xlabel("Niveau de barriere (%)")
    ax3.set_ylabel("P(barriere touchee) (%)")
    ax3.legend(fontsize=8, facecolor="#1a1d27", labelcolor="white")
    ax3.grid(True, alpha=0.15)

    fig.suptitle(
        "Comparaison Athena — Indice vs Panier personnalise\n"
        "Heston calibre sur donnees reelles",
        color="white", fontsize=11, y=1.02
    )

    out = "athena_comparaison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Graphique comparaison sauvegarde : {out}")


# ─────────────────────────────────────────────
# POINT D'ENTREE
# ─────────────────────────────────────────────
if __name__ == "__main__":

    START  = "2015-01-01"
    END    = "2025-01-01"
    M_SIM  = 10_000
    STEPS  = 52
    params = ATHENA_DEFAULT
    T      = params["T"]

    results_dict = {}
    S_dict       = {}

    # ════════════════════════════════════
    # MODE A — Indice : EuroStoxx 50
    # ════════════════════════════════════
    print("=" * 65)
    print("  MODE A — Indice : EuroStoxx 50")
    print("=" * 65)

    prices_idx       = get_index("^STOXX50E", start=START, end=END)
    var_idx, lr_idx  = compute_realized_variance(prices_idx, window=21)
    heston_idx       = calibrate_heston(var_idx, lr_idx)
    print_calibration(heston_idx, label="(EuroStoxx 50)")

    S_idx, _, _ = simulate_heston(heston_idx, T=T, M=M_SIM, steps_per_year=STEPS)
    res_idx     = pricer_athena(S_idx, params=params, steps_per_year=STEPS)
    print_stats_athena(res_idx, "EuroStoxx 50", heston_idx, params)
    plot_athena_single(S_idx, res_idx, heston_idx, label="EuroStoxx 50", params=params)

    results_dict["EuroStoxx 50"] = res_idx
    S_dict["EuroStoxx 50"]       = S_idx

    # ════════════════════════════════════
    # MODE B — Panier : LVMH / Total / Airbus
    # ════════════════════════════════════
    print("\n" + "=" * 65)
    print("  MODE B — Panier : LVMH / Total / Airbus")
    print("=" * 65)

    tickers = ["MC.PA", "TTE.PA", "AIR.PA"]
    weights  = [0.40,    0.30,     0.30]

    prices_bsk       = get_basket(tickers, weights, start=START, end=END)
    var_bsk, lr_bsk  = compute_realized_variance(prices_bsk, window=21)
    heston_bsk       = calibrate_heston(var_bsk, lr_bsk)
    print_calibration(heston_bsk, label="(Panier LVMH/Total/Airbus)")

    S_bsk, _, _ = simulate_heston(heston_bsk, T=T, M=M_SIM, steps_per_year=STEPS)
    res_bsk     = pricer_athena(S_bsk, params=params, steps_per_year=STEPS)
    print_stats_athena(res_bsk, "Panier LVMH-Total-Airbus", heston_bsk, params)
    plot_athena_single(S_bsk, res_bsk, heston_bsk, label="Panier LVMH-Total-Airbus", params=params)

    results_dict["Panier LVMH-Total-Airbus"] = res_bsk
    S_dict["Panier LVMH-Total-Airbus"]       = S_bsk

    # ════════════════════════════════════
    # COMPARAISON
    # ════════════════════════════════════
    print("\nGeneration du graphique de comparaison...")
    plot_comparison(results_dict, S_dict, params=params)

    print("\n Bloc 2 termine.")