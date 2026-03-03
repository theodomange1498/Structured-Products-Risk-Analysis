"""
BLOC 1 — Simulateur de Heston avec calibration sur données réelles
===================================================================
Deux modes disponibles :

  MODE A — Indice unique
    ticker = "^STOXX50E"  (EuroStoxx 50)
    ticker = "^FCHI"      (CAC 40)
    ticker = "^GSPC"      (S&P 500)

  MODE B — Panier d'actions personnalisé
    tickers = ["MC.PA", "TTE.PA", "AIR.PA"]   (LVMH, Total, Airbus)
    weights  = [0.4, 0.3, 0.3]
    Le panier est normalisé à 100 au départ.

Pipeline :
  1. Téléchargement des prix via yfinance
  2. Calcul de la volatilité réalisée (fenêtre glissante 21 jours)
  3. Calibration des paramètres Heston par moindres carrés
     sur la série de variances réalisées
  4. Simulation Monte Carlo avec ces paramètres calibrés
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# ÉTAPE 1 — TÉLÉCHARGEMENT ET CONSTRUCTION DU SOUS-JACENT
# ─────────────────────────────────────────────

def get_index(ticker, start, end):
    """
    Télécharge un indice unique et retourne la série de prix normalisée à 100.
    """
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    prices = data["Close"].dropna()
    if hasattr(prices, 'squeeze'):
        prices = prices.squeeze()
    prices = prices / prices.iloc[0] * 100
    print(f"  Indice {ticker} : {len(prices)} jours du {prices.index[0].date()} au {prices.index[-1].date()}")
    return prices


def get_basket(tickers, weights, start, end):
    """
    Télécharge un panier d'actions, applique les poids et retourne
    la série de prix du panier normalisée à 100.

    Paramètres :
        tickers : liste de tickers Yahoo Finance (ex: ["MC.PA", "TTE.PA"])
        weights : liste de poids (doit sommer à 1)
        start, end : dates au format "YYYY-MM-DD"
    """
    weights = np.array(weights)
    weights = weights / weights.sum()

    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)["Close"]
    data = data.dropna()

    # Normalise chaque action à 100 au départ
    data_norm = data / data.iloc[0] * 100

    # Panier = somme pondérée
    basket = (data_norm * weights).sum(axis=1)
    basket = basket / basket.iloc[0] * 100

    print(f"  Panier {tickers} : {len(basket)} jours")
    for t, w in zip(tickers, weights):
        print(f"    {t} : {w*100:.0f}%")
    return basket


# ─────────────────────────────────────────────
# ÉTAPE 2 — VOLATILITÉ RÉALISÉE
# ─────────────────────────────────────────────

def compute_realized_variance(prices, window=21):
    """
    Calcule la variance réalisée journalière via une fenêtre glissante.

    Méthode : variance des log-rendements sur 'window' jours, annualisée.
    window=21 ≈ 1 mois de trading — standard pour estimer la vol court terme.

    Retourne :
        var_series  : pd.Series de variances annualisées
        log_returns : pd.Series des log-rendements journaliers
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    # Variance glissante annualisée (×252 car journalier)
    var_series = log_returns.rolling(window).var() * 252
    var_series = var_series.dropna()
    return var_series, log_returns


# ─────────────────────────────────────────────
# ÉTAPE 3 — CALIBRATION HESTON
# ─────────────────────────────────────────────

def calibrate_heston(var_series, log_returns):
    """
    Calibre les paramètres Heston (κ, θ, ξ, ρ, v0) sur la série
    de variances réalisées historiques.

    Idée : dans Heston, la variance suit dv = κ(θ-v)dt + ξ√v dW₂
    On estime κ et θ par régression AR(1) sur var_series,
    puis ξ par l'écart-type des résidus,
    et ρ par la corrélation entre log-rendements et Δvariance.

    C'est une calibration "moments" — plus simple qu'une calibration
    sur surface de vol implicite (qui nécessite des prix d'options),
    mais suffisante et économiquement solide pour ce projet.
    """
    v = var_series.values
    dt = 1 / 252  # pas journalier

    # ── Estimation de κ et θ par régression AR(1) ──
    # dv ≈ κ(θ - v)dt  →  v[t+1] - v[t] = κθ·dt - κ·v[t]·dt
    # On pose : y = v[t+1] - v[t],  x = v[t]
    # Régression : y = a + b·x  →  b = -κ·dt,  a = κθ·dt
    y = np.diff(v)
    x = v[:-1]

    x_mean, y_mean = x.mean(), y.mean()
    b = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    a = y_mean - b * x_mean

    kappa = max(-b / dt, 0.1)
    theta = max(a / (kappa * dt), 0.001)

    # ── Estimation de ξ (vol de la vol) ──
    residuals = y - (a + b * x)
    v_bar = np.sqrt(np.maximum(x, 1e-6))
    xi = np.std(residuals) / (np.mean(v_bar) * np.sqrt(dt))
    xi = np.clip(xi, 0.05, 2.0)

    # ── Estimation de ρ (levier) ──
    # Corrélation entre log-returns et variation de variance
    lr = log_returns.values
    n = min(len(lr), len(y))
    lr_aligned = lr[-n:]
    dv_aligned = y[-n:]
    corr_matrix = np.corrcoef(lr_aligned, dv_aligned)
    rho = np.clip(corr_matrix[0, 1], -0.98, -0.01)

    # ── v0 : variance initiale = dernière valeur observée ──
    v0 = float(v[-1])
    v0 = np.clip(v0, 0.005, 0.5)

    params = {
        "v0":    round(v0, 4),
        "kappa": round(float(kappa), 3),
        "theta": round(float(theta), 4),
        "xi":    round(float(xi), 3),
        "rho":   round(float(rho), 3),
    }
    return params


def print_calibration(params, label=""):
    """Affiche les paramètres calibrés avec leur interprétation."""
    print(f"\n  Paramètres Heston calibrés {label}:")
    print(f"    v0    = {params['v0']:.4f}  → vol initiale   = {np.sqrt(params['v0'])*100:.1f}%")
    print(f"    θ     = {params['theta']:.4f}  → vol long terme = {np.sqrt(params['theta'])*100:.1f}%")
    print(f"    κ     = {params['kappa']:.3f}  → vitesse mean reversion")
    print(f"    ξ     = {params['xi']:.3f}  → vol de la vol")
    print(f"    ρ     = {params['rho']:.3f}  → leverage effect")


# ─────────────────────────────────────────────
# ÉTAPE 4 — SIMULATION HESTON
# ─────────────────────────────────────────────

def simulate_heston(
    heston_params,
    S0=100.0,
    mu=0.02,
    T=8,
    steps_per_year=52,
    M=10_000,
    seed=42,
):
    """
    Simule M trajectoires du modèle de Heston.

    Paramètres :
        heston_params : dict issu de calibrate_heston()
        S0            : niveau initial du sous-jacent (base 100)
        mu            : drift annuel (taux sans risque ou rendement espéré)
        T             : maturité en années
        steps_per_year: pas de temps par an (52=hebdo, 252=quotidien)
        M             : nombre de simulations
        seed          : graine aléatoire (reproductibilité)

    Retourne :
        S : array (M, N+1) — trajectoires de prix
        V : array (M, N+1) — trajectoires de variance instantanée
        t : array (N+1,)   — axe temporel en années
    """
    rng = np.random.default_rng(seed)

    v0    = heston_params["v0"]
    kappa = heston_params["kappa"]
    theta = heston_params["theta"]
    xi    = heston_params["xi"]
    rho   = heston_params["rho"]

    N  = T * steps_per_year
    dt = 1.0 / steps_per_year
    t  = np.linspace(0, T, N + 1)

    S = np.zeros((M, N + 1))
    V = np.zeros((M, N + 1))
    S[:, 0] = S0
    V[:, 0] = v0

    for i in range(N):
        Z1      = rng.standard_normal(M)
        Z_indep = rng.standard_normal(M)
        Z2      = rho * Z1 + np.sqrt(1 - rho**2) * Z_indep

        v_curr = np.maximum(V[:, i], 0.0)
        sqrt_v = np.sqrt(v_curr)

        # Variance (schéma full truncation — garantit v > 0)
        V[:, i+1] = np.maximum(
            v_curr + kappa * (theta - v_curr) * dt + xi * sqrt_v * np.sqrt(dt) * Z2,
            0.0
        )

        # Prix (log-normal avec vol stochastique)
        S[:, i+1] = S[:, i] * np.exp(
            (mu - 0.5 * v_curr) * dt + sqrt_v * np.sqrt(dt) * Z1
        )

    return S, V, t


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

def plot_calibration_and_simulation(prices, var_series, S, V, t, params, label=""):
    """
    3 panels :
      - Haut gauche  : prix historique + vol réalisée
      - Haut droite  : trajectoires simulées (percentiles)
      - Bas          : distribution finale des prix (KDE)
    """
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1  = fig.add_subplot(gs[0, 0])
    ax1b = ax1.twinx()
    ax2  = fig.add_subplot(gs[0, 1])
    ax3  = fig.add_subplot(gs[1, :])

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
    ax1b.tick_params(colors="#ff6b6b")
    ax1b.yaxis.label.set_color("#ff6b6b")

    # ── Historique ──
    ax1.plot(prices.index, prices.values, color="#4a9eff", linewidth=1.5)
    ax1b.plot(var_series.index, np.sqrt(var_series.values) * 100,
              color="#ff6b6b", linewidth=1, alpha=0.7)
    ax1.set_title(f"Données historiques — {label}", fontsize=11)
    ax1.set_ylabel("Prix (base 100)", color="#4a9eff")
    ax1b.set_ylabel("Vol réalisée (%)", color="#ff6b6b")
    ax1.set_facecolor("#1a1d27")

    # ── Trajectoires simulées ──
    idx_plot = np.random.choice(S.shape[0], 80, replace=False)
    for i in idx_plot:
        ax2.plot(t, S[i], color="#4a9eff", alpha=0.08, linewidth=0.6)
    p10 = np.percentile(S, 10, axis=0)
    p50 = np.percentile(S, 50, axis=0)
    p90 = np.percentile(S, 90, axis=0)
    ax2.fill_between(t, p10, p90, color="#4a9eff", alpha=0.2, label="P10-P90")
    ax2.plot(t, p50, color="white", linewidth=2, label="Médiane")
    ax2.axhline(100, color="#888", linestyle="--", linewidth=0.8)
    ax2.set_title("Trajectoires simulées (Heston calibré)", fontsize=11)
    ax2.set_xlabel("Années")
    ax2.set_ylabel("Indice (base 100)")
    ax2.legend(fontsize=8, facecolor="#1a1d27", labelcolor="white")

    # ── Distribution finale (KDE) ──
    S_final = S[:, -1]
    kde = gaussian_kde(S_final, bw_method=0.15)
    x_range = np.linspace(S_final.min(), S_final.max(), 500)
    ax3.plot(x_range, kde(x_range), color="#4a9eff", linewidth=2.5)
    ax3.fill_between(x_range, kde(x_range), alpha=0.2, color="#4a9eff")
    ax3.axvline(100, color="white", linestyle="--", linewidth=1.2, label="Capital initial")
    ax3.axvline(60,  color="#ff6b6b", linestyle="--", linewidth=1.2, label="Barrière 60%")
    ax3.axvline(np.percentile(S_final, 10), color="#ffcc00", linestyle=":",
                linewidth=1.2, label=f"P10 = {np.percentile(S_final,10):.0f}")
    ax3.set_title(f"Distribution du prix final à {t[-1]:.0f} ans", fontsize=11)
    ax3.set_xlabel("Prix final (base 100)")
    ax3.set_ylabel("Densité")
    ax3.legend(fontsize=9, facecolor="#1a1d27", labelcolor="white")

    v0_pct = np.sqrt(params['v0']) * 100
    th_pct = np.sqrt(params['theta']) * 100
    fig.suptitle(
        f"Heston calibré sur données réelles — {label}\n"
        f"v₀={v0_pct:.1f}%  θ={th_pct:.1f}%  "
        f"κ={params['kappa']}  ξ={params['xi']}  ρ={params['rho']}"
        f"  |  {S.shape[0]:,} simulations sur {t[-1]:.0f} ans",
        color="white", fontsize=10, y=1.01
    )

    out = f"heston_calibre_{label.replace(' ','_').replace('/','_')}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Graphique sauvegarde : {out}")


# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Fenêtre historique ──
    START = "2015-01-01"
    END   = "2025-01-01"
    T_SIM = 8
    M_SIM = 10_000

    # ════════════════════════════════════════
    # MODE A — Indice unique : EuroStoxx 50
    # ════════════════════════════════════════
    print("=" * 60)
    print("  MODE A — Indice : EuroStoxx 50")
    print("=" * 60)

    prices_index           = get_index("^STOXX50E", start=START, end=END)
    var_index, lr_index    = compute_realized_variance(prices_index, window=21)
    params_index           = calibrate_heston(var_index, lr_index)
    print_calibration(params_index, label="(EuroStoxx 50)")

    S_idx, V_idx, t = simulate_heston(params_index, T=T_SIM, M=M_SIM)
    plot_calibration_and_simulation(
        prices_index, var_index, S_idx, V_idx, t,
        params_index, label="EuroStoxx 50"
    )

    # ════════════════════════════════════════
    # MODE B — Panier personnalisé
    # ════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  MODE B — Panier : LVMH / Total / Airbus")
    print("=" * 60)

    tickers = ["MC.PA", "TTE.PA", "AIR.PA"]
    weights  = [0.40, 0.30, 0.30]

    prices_basket          = get_basket(tickers, weights, start=START, end=END)
    var_basket, lr_basket  = compute_realized_variance(prices_basket, window=21)
    params_basket          = calibrate_heston(var_basket, lr_basket)
    print_calibration(params_basket, label="(Panier LVMH/Total/Airbus)")

    S_bsk, V_bsk, t = simulate_heston(params_basket, T=T_SIM, M=M_SIM)
    plot_calibration_and_simulation(
        prices_basket, var_basket, S_bsk, V_bsk, t,
        params_basket, label="Panier LVMH-Total-Airbus"
    )

    print("\n✅ Bloc 1 terminé.")
    print(f"   params_index  = {params_index}")
    print(f"   params_basket = {params_basket}")
    