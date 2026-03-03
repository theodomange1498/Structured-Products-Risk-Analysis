"""
BLOC 3 — Interface Streamlit : Athena Pricer
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
import sys, os, io
from datetime import date
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import Image as RLImage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heston_simulator import (
    get_index, get_basket,
    compute_realized_variance, calibrate_heston,
    simulate_heston, print_calibration
)
from athena_payoff import pricer_athena

st.set_page_config(
    page_title="Structured Products Risk Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; background-color: #080c14; color: #e2e8f0; }
section[data-testid="stSidebar"] { background: #0d1220; border-right: 1px solid #1e293b; }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
.metric-card { background: #0d1220; border: 1px solid #1e293b; border-radius: 8px; padding: 16px 20px; text-align: center; }
.metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 2rem; font-weight: 600; line-height: 1.1; }
.metric-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }
.metric-red   { color: #f87171; }
.metric-green { color: #34d399; }
.metric-blue  { color: #60a5fa; }
.metric-gold  { color: #fbbf24; }
.main-title { font-family: 'IBM Plex Mono', monospace; font-size: 2.2rem; font-weight: 600; color: #f1f5f9; margin-bottom: 0; }
.main-subtitle { font-size: 0.85rem; color: #475569; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 2rem; }
.section-header { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #3b82f6; text-transform: uppercase; letter-spacing: 0.15em; border-bottom: 1px solid #1e293b; padding-bottom: 6px; margin: 2rem 0 1rem 0; }
.fiche { background: #0d1220; border: 1px solid #1e293b; border-radius: 8px; padding: 24px; font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; line-height: 1.8; }
.fiche-title { font-size: 1rem; font-weight: 600; color: #60a5fa; margin-bottom: 12px; }
.fiche-row { display: flex; justify-content: space-between; border-bottom: 1px solid #1e293b; padding: 4px 0; }
.fiche-key { color: #64748b; }
.fiche-val { color: #e2e8f0; font-weight: 600; }
div.stButton > button { background: #1d4ed8; color: white; border: none; border-radius: 6px; font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; font-weight: 600; padding: 10px 24px; width: 100%; }
div.stButton > button:hover { background: #2563eb; }
button[data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78rem !important; text-transform: uppercase !important; }
div[data-testid="stExpander"] { background: #0d1220; border: 1px solid #1e293b; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

INDICES_CATALOGUE = {
    "Europe": {
        "EuroStoxx 50": "^STOXX50E",
        "CAC 40":       "^FCHI",
        "DAX":          "^GDAXI",
        "FTSE 100":     "^FTSE",
        "SMI (Suisse)": "^SSMI",
        "IBEX 35":      "^IBEX",
        "AEX":          "^AEX",
    },
    "Etats-Unis": {
        "S&P 500":      "^GSPC",
        "Nasdaq 100":   "^NDX",
        "Dow Jones":    "^DJI",
        "Russell 2000": "^RUT",
    },
    "Asie & Monde": {
        "Nikkei 225":   "^N225",
        "Hang Seng":    "^HSI",
        "MSCI World":   "URTH",
        "MSCI EM":      "EEM",
    },
}

ACTIONS_CATALOGUE = {
    "France": {
        "LVMH":         "MC.PA",
        "TotalEnergies":"TTE.PA",
        "Airbus":       "AIR.PA",
        "Hermes":       "RMS.PA",
        "Sanofi":       "SAN.PA",
        "BNP Paribas":  "BNP.PA",
        "Schneider":    "SU.PA",
        "L'Oreal":      "OR.PA",
    },
    "Europe": {
        "ASML":         "ASML.AS",
        "SAP":          "SAP.DE",
        "Siemens":      "SIE.DE",
        "BASF":         "BAS.DE",
        "Allianz":      "ALV.DE",
    },
    "Etats-Unis": {
        "Apple":        "AAPL",
        "Microsoft":    "MSFT",
        "Amazon":       "AMZN",
        "Alphabet":     "GOOGL",
        "NVIDIA":       "NVDA",
        "Tesla":        "TSLA",
        "JPMorgan":     "JPM",
        "ExxonMobil":   "XOM",
    },
}

BG      = "#080c14"
CARD_BG = "#0d1220"
BORDER  = "#1e293b"
BLUE    = "#3b82f6"
RED     = "#f87171"
GREEN   = "#34d399"
GOLD    = "#fbbf24"
TEXT    = "#e2e8f0"
MUTED   = "#475569"

# TEXT partout pour que les axes/chiffres soient bien visibles
PLT_STYLE = {
    "figure.facecolor": BG,    "axes.facecolor": CARD_BG,
    "axes.edgecolor":  BORDER, "axes.labelcolor": TEXT,
    "xtick.color":     TEXT,   "ytick.color":    TEXT,
    "text.color":      TEXT,   "grid.color":     BORDER,
    "grid.alpha":      0.5,
}

def style_ax(ax):
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

def fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_payoff_kde(payoffs, params, label):
    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 3.8))
        style_ax(ax)
        kde = gaussian_kde(payoffs, bw_method=0.15)
        x   = np.linspace(max(0, payoffs.min()), payoffs.max(), 600)
        y   = kde(x)
        ax.plot(x, y, color=BLUE, linewidth=2.2)
        ax.fill_between(x, y, alpha=0.18, color=BLUE)
        bp = params["barriere_protect"] * 100
        x_loss = x[x < bp]
        if len(x_loss):
            ax.fill_between(x_loss, kde(x_loss), alpha=0.35, color=RED)
        ax.axvline(100, color=TEXT, lw=1.2, ls="--", label="Capital initial")
        ax.axvline(bp,  color=RED,  lw=1.2, ls="--", label=f"Barriere {bp:.0f}%")
        ax.axvline(np.percentile(payoffs, 10), color=GOLD, lw=1, ls=":",
                   label=f"P10 = {np.percentile(payoffs,10):.1f}")
        ax.set_title("Distribution des payoffs", fontsize=10, pad=10)
        ax.set_xlabel("Payoff final  (investi = 100)")
        ax.set_ylabel("Densite")
        ax.legend(fontsize=7.5, facecolor=CARD_BG, labelcolor=TEXT, framealpha=0.9, edgecolor=BORDER)
        fig.tight_layout()
    return fig_to_buf(fig)

def plot_sensitivity(S, params, label):
    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 3.8))
        style_ax(ax)
        niveaux   = np.arange(0.40, 0.86, 0.025)
        p_barrier = [np.mean((S < 100*niv).any(axis=1))*100 for niv in niveaux]
        ax.plot(niveaux*100, p_barrier, color=BLUE, lw=2.2, marker="o", markersize=3.5)
        ax.fill_between(niveaux*100, p_barrier, alpha=0.14, color=BLUE)
        bp  = params["barriere_protect"] * 100
        p_b = np.mean((S < bp).any(axis=1)) * 100
        ax.axvline(bp, color=RED, lw=1.4, ls="--", label=f"Barriere {bp:.0f}% -> {p_b:.1f}%")
        ax.scatter([bp], [p_b], color=RED, zorder=6, s=50)
        ax.set_title("Sensibilite P(barriere touchee) vs niveau barriere", fontsize=10, pad=10)
        ax.set_xlabel("Niveau de barriere (%)")
        ax.set_ylabel("P(barriere touchee) (%)")
        ax.legend(fontsize=7.5, facecolor=CARD_BG, labelcolor=TEXT, framealpha=0.9, edgecolor=BORDER)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
    return fig_to_buf(fig)

def plot_recall_bars(recall_proba, T, label):
    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 3.8))
        style_ax(ax)
        rp   = recall_proba * 100
        x    = np.arange(1, T+1)
        cols = [BLUE if v == rp.max() else "#1e3a5f" for v in rp]
        bars = ax.bar(x, rp, color=cols, alpha=0.9, width=0.6, zorder=3)
        for bar, val in zip(bars, rp):
            if val > 0.5:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                        f"{val:.1f}%", ha="center", va="bottom", color=TEXT, fontsize=8)
        ax.set_title("Probabilite de rappel par annee", fontsize=10, pad=10)
        ax.set_xlabel("Annee de constatation")
        ax.set_ylabel("Probabilite (%)")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.2, axis="y", zorder=0)
        fig.tight_layout()
    return fig_to_buf(fig)

def plot_cumul_barrier(S, params, label):
    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 3.8))
        style_ax(ax)
        T              = params["T"]
        steps          = S.shape[1] - 1
        spy            = steps // T
        niveau_protect = 100 * params["barriere_protect"]
        pas            = max(spy // 4, 1)
        cumul = np.array([
            np.mean((S[:, :i+1] < niveau_protect).any(axis=1))
            for i in range(0, S.shape[1], pas)
        ]) * 100
        t_trim = np.linspace(0, T, len(cumul))
        ax.plot(t_trim, cumul, color=BLUE, lw=2.2)
        ax.fill_between(t_trim, cumul, alpha=0.14, color=BLUE)
        ax.scatter([t_trim[-1]], [cumul[-1]], color=RED, zorder=6, s=50,
                   label=f"A maturite : {cumul[-1]:.1f}%")
        ax.set_title("P(barriere touchee) cumulee dans le temps", fontsize=10, pad=10)
        ax.set_xlabel("Annees depuis emission")
        ax.set_ylabel("P(barriere touchee au moins une fois) (%)")
        ax.legend(fontsize=7.5, facecolor=CARD_BG, labelcolor=TEXT, framealpha=0.9, edgecolor=BORDER)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
    return fig_to_buf(fig)

def plot_comparison(res_a, S_a, label_a, res_b, S_b, label_b, params):
    with plt.rc_context(PLT_STYLE):
        fig = plt.figure(figsize=(14, 4.5))
        fig.patch.set_facecolor(BG)
        gs   = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
        axes = [fig.add_subplot(gs[0, j]) for j in range(3)]
        for ax in axes: style_ax(ax)
        ax1, ax2, ax3 = axes
        cls  = [BLUE, RED]
        lbls = [label_a, label_b]
        for i, (res, lbl) in enumerate(zip([res_a, res_b], lbls)):
            p   = res["payoffs"]
            kde = gaussian_kde(p, bw_method=0.15)
            x   = np.linspace(max(0, p.min()), p.max(), 500)
            ax1.plot(x, kde(x), color=cls[i], lw=2, label=lbl)
            ax1.fill_between(x, kde(x), alpha=0.10, color=cls[i])
        ax1.axvline(100, color=TEXT, lw=1, ls="--", alpha=0.5)
        ax1.set_title("Distribution des payoffs", fontsize=9.5, pad=8)
        ax1.set_xlabel("Payoff final")
        ax1.legend(fontsize=7.5, facecolor=CARD_BG, labelcolor=TEXT, edgecolor=BORDER)
        metriques = ["P_barriere", "P_rappel", "P_perte_maturite"]
        labels_m  = ["P(barriere)", "P(rappel)", "P(perte mat.)"]
        x_pos = np.arange(len(metriques))
        width = 0.32
        for i, (res, lbl) in enumerate(zip([res_a, res_b], lbls)):
            d       = res["details"]
            valeurs = [d[m]*100 for m in metriques]
            offset  = (i-0.5)*width
            bars    = ax2.bar(x_pos+offset, valeurs, width=width*0.88, color=cls[i], alpha=0.88, label=lbl)
            for bar, val in zip(bars, valeurs):
                ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                         f"{val:.1f}%", ha="center", va="bottom", color=TEXT, fontsize=7)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels_m, fontsize=8)
        ax2.set_title("Metriques cles", fontsize=9.5, pad=8)
        ax2.set_ylabel("Probabilite (%)")
        ax2.legend(fontsize=7.5, facecolor=CARD_BG, labelcolor=TEXT, edgecolor=BORDER)
        ax2.grid(True, alpha=0.2, axis="y")
        niveaux = np.arange(0.40, 0.86, 0.025)
        for i, (S, lbl) in enumerate(zip([S_a, S_b], lbls)):
            pb = [np.mean((S < 100*niv).any(axis=1))*100 for niv in niveaux]
            ax3.plot(niveaux*100, pb, color=cls[i], lw=2, marker="o", markersize=3, label=lbl)
        ax3.axvline(params["barriere_protect"]*100, color=TEXT, lw=1, ls="--", alpha=0.4)
        ax3.set_title("Sensibilite P(barriere)", fontsize=9.5, pad=8)
        ax3.set_xlabel("Niveau barriere (%)")
        ax3.set_ylabel("P(barriere touchee) (%)")
        ax3.legend(fontsize=7.5, facecolor=CARD_BG, labelcolor=TEXT, edgecolor=BORDER)
        ax3.grid(True, alpha=0.2)
        fig.suptitle(f"Comparaison  {label_a}  vs  {label_b}", color=TEXT, fontsize=10, y=1.02)
        fig.tight_layout()
    return fig_to_buf(fig)

def generate_pdf(label, params, heston_params, details, recall_proba,
                 buf_kde, buf_sensitivity, buf_recalls, buf_cumul):
    pdf_buf = io.BytesIO()
    C_DARK  = rl_colors.HexColor("#080c14")
    C_CARD  = rl_colors.HexColor("#0d1220")
    C_BORD  = rl_colors.HexColor("#1e293b")
    C_BLUE  = rl_colors.HexColor("#3b82f6")
    C_RED   = rl_colors.HexColor("#f87171")
    C_GREEN = rl_colors.HexColor("#34d399")
    C_GOLD  = rl_colors.HexColor("#fbbf24")
    C_TEXT  = rl_colors.HexColor("#e2e8f0")
    C_MUTED = rl_colors.HexColor("#64748b")
    W, H = A4
    doc  = SimpleDocTemplate(pdf_buf, pagesize=A4,
                              leftMargin=18*mm, rightMargin=18*mm,
                              topMargin=16*mm,  bottomMargin=16*mm)
    def S(name, **kw):
        base = {"fontName":"Helvetica","fontSize":9,"textColor":C_TEXT,"leading":14}
        base.update(kw)
        return ParagraphStyle(name, **base)
    s_title  = S("t",  fontName="Helvetica-Bold", fontSize=20, textColor=C_BLUE,  leading=24, spaceAfter=2)
    s_sub    = S("s",  fontSize=8,  textColor=C_MUTED, leading=12, spaceAfter=10)
    s_sec    = S("sc", fontName="Helvetica-Bold", fontSize=7, textColor=C_BLUE, leading=10, spaceBefore=12, spaceAfter=4)
    s_body   = S("b",  fontSize=8.5, leading=13)
    s_small  = S("sm", fontSize=7,   textColor=C_MUTED, leading=10)
    s_bold_r = S("br", fontName="Helvetica-Bold", fontSize=8.5, alignment=TA_RIGHT)
    s_red    = S("rd", fontName="Helvetica-Bold", fontSize=8.5, textColor=C_RED,   alignment=TA_RIGHT)
    s_green  = S("gr", fontName="Helvetica-Bold", fontSize=8.5, textColor=C_GREEN, alignment=TA_RIGHT)
    s_gold   = S("gd", fontName="Helvetica-Bold", fontSize=8.5, textColor=C_GOLD,  alignment=TA_RIGHT)
    s_blue_r = S("bl", fontName="Helvetica-Bold", fontSize=8.5, textColor=C_BLUE,  alignment=TA_RIGHT)
    s_hdr    = S("hd", fontName="Helvetica-Bold", fontSize=7, textColor=C_MUTED)
    s_hdr_c  = S("hc", fontName="Helvetica-Bold", fontSize=7, textColor=C_MUTED, alignment=TA_CENTER)
    v0_pct = np.sqrt(heston_params["v0"]) * 100
    th_pct = np.sqrt(heston_params["theta"]) * 100
    story = []
    story.append(Paragraph("Structured Products Risk Analysis", s_title))
    story.append(Paragraph(f"Structured Products Risk Analysis  .  {date.today().strftime('%d %B %Y')}", s_sub))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORD, spaceAfter=10))
    story.append(Paragraph("TERM SHEET  &amp;  RESULTATS", s_sec))
    col_w = [(W-36*mm)*x for x in [0.27, 0.18, 0.05, 0.31, 0.19]]
    ts = [
        [Paragraph("<b>PRODUIT</b>", s_hdr), Paragraph("<b>VALEUR</b>", S("h", fontName="Helvetica-Bold", fontSize=7, textColor=C_MUTED, alignment=TA_RIGHT)), Paragraph(""), Paragraph("<b>SIMULATION</b>", s_hdr), Paragraph("<b>RESULTAT</b>", S("h2", fontName="Helvetica-Bold", fontSize=7, textColor=C_MUTED, alignment=TA_RIGHT))],
        [Paragraph("Sous-jacent", s_body),        Paragraph(f"<b>{label}</b>", s_bold_r),                             Paragraph(""), Paragraph("P(barriere touchee)", s_body),  Paragraph(f"<b>{details['P_barriere']*100:.1f}%</b>", s_red)],
        [Paragraph("Maturite max", s_body),        Paragraph(f"<b>{params['T']} ans</b>", s_bold_r),                   Paragraph(""), Paragraph("P(rappel anticipe)", s_body),   Paragraph(f"<b>{details['P_rappel']*100:.1f}%</b>", s_green)],
        [Paragraph("Coupon annuel", s_body),       Paragraph(f"<b>{params['coupon_annuel']*100:.1f}%</b>", s_bold_r),  Paragraph(""), Paragraph("P(perte a maturite)", s_body),  Paragraph(f"<b>{details['P_perte_maturite']*100:.1f}%</b>", s_red)],
        [Paragraph("Barriere rappel", s_body),     Paragraph(f"<b>{params['barriere_rappel']*100:.0f}%</b>", s_bold_r),Paragraph(""), Paragraph("Payoff moyen", s_body),         Paragraph(f"<b>{details['payoff_moyen']:.1f}</b>", s_gold)],
        [Paragraph("Barriere protection", s_body), Paragraph(f"<b>{params['barriere_protect']*100:.0f}%</b>", s_bold_r),Paragraph(""),Paragraph("Payoff P10 / P90", s_body),    Paragraph(f"<b>{details['payoff_p10']:.1f} / {details['payoff_p90']:.1f}</b>", s_bold_r)],
        [Paragraph("Nominal", s_body),             Paragraph(f"<b>{params['nominal']:.0f}</b>", s_bold_r),             Paragraph(""), Paragraph("Duree de vie moy.", s_body),    Paragraph(f"<b>{details['duree_vie_moyenne']:.1f} ans</b>", s_bold_r)],
        [Paragraph("", s_body),                    Paragraph("", s_bold_r),                                            Paragraph(""), Paragraph("Taux annualise moy.", s_body),  Paragraph(f"<b>{details['taux_actualise']:.2f}%</b>", s_blue_r)],
    ]
    ts_tbl = Table(ts, colWidths=col_w)
    ts_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  C_CARD),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_DARK, C_CARD]),
        ("LINEBELOW",     (0,0),(-1,0),  0.5, C_BORD),
        ("LINEAFTER",     (1,0),(2,-1),  0.5, C_BORD),
        ("TOPPADDING",    (0,0),(-1,-1), 5), ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",   (0,0),(-1,-1), 8), ("RIGHTPADDING", (0,0),(-1,-1),8),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]))
    story.append(ts_tbl)
    story.append(Spacer(1, 10))
    story.append(Paragraph("PROBABILITES DE RAPPEL PAR ANNEE", s_sec))
    T = params["T"]
    r_hdrs = [Paragraph(f"<b>An {k}</b>", s_hdr_c) for k in range(1, T+1)]
    r_vals  = [Paragraph(f"<b>{p*100:.1f}%</b>", S("rv", fontName="Helvetica-Bold", fontSize=8.5, textColor=C_BLUE if p==recall_proba.max() else C_TEXT, alignment=TA_CENTER)) for p in recall_proba]
    r_tbl = Table([r_hdrs, r_vals], colWidths=[(W-36*mm)/T]*T)
    r_tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),C_CARD), ("BACKGROUND",(0,1),(-1,1),C_DARK),
        ("LINEBELOW",(0,0),(-1,0),0.5,C_BORD),
        ("TOPPADDING",(0,0),(-1,-1),5), ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
    ]))
    story.append(r_tbl)
    story.append(Spacer(1, 10))
    story.append(Paragraph("MODELE — HESTON CALIBRE SUR DONNEES REELLES", s_sec))
    h_data = [
        [Paragraph("<b>Parametre</b>", s_hdr), Paragraph("<b>Valeur</b>", s_hdr), Paragraph("<b>Interpretation</b>", s_hdr)],
        [Paragraph("v0 (vol initiale)",     s_body), Paragraph(f"<b>{v0_pct:.1f}%</b>", S("hv1",fontName="Helvetica-Bold",fontSize=8.5,textColor=C_BLUE)), Paragraph("Volatilite au moment de l'emission", s_small)],
        [Paragraph("theta (vol long terme)",s_body), Paragraph(f"<b>{th_pct:.1f}%</b>", S("hv2",fontName="Helvetica-Bold",fontSize=8.5,textColor=C_BLUE)), Paragraph("Niveau vers lequel la vol converge",  s_small)],
        [Paragraph("kappa (mean reversion)",s_body), Paragraph(f"<b>{heston_params['kappa']}</b>", S("hv3",fontName="Helvetica-Bold",fontSize=8.5,textColor=C_TEXT)), Paragraph("Vitesse de retour a la moyenne", s_small)],
        [Paragraph("xi (vol de la vol)",    s_body), Paragraph(f"<b>{heston_params['xi']}</b>",    S("hv4",fontName="Helvetica-Bold",fontSize=8.5,textColor=C_TEXT)), Paragraph("Amplitude des chocs de volatilite", s_small)],
        [Paragraph("rho (leverage effect)", s_body), Paragraph(f"<b>{heston_params['rho']}</b>",   S("hv5",fontName="Helvetica-Bold",fontSize=8.5,textColor=C_TEXT)), Paragraph("Correlation rendements / volatilite", s_small)],
    ]
    h_tbl = Table(h_data, colWidths=[(W-36*mm)*x for x in [0.28, 0.15, 0.57]])
    h_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  C_CARD),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_DARK, C_CARD]),
        ("LINEBELOW",     (0,0),(-1,0),  0.5, C_BORD),
        ("TOPPADDING",    (0,0),(-1,-1), 4), ("BOTTOMPADDING",(0,0),(-1,-1),4),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
    ]))
    story.append(h_tbl)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORD))
    story.append(Paragraph("Document genere par Structured Products Risk Analysis — Modele de Heston calibre sur donnees historiques. Ce document est fourni a titre informatif et ne constitue pas un conseil en investissement.", S("disc", fontSize=6.5, textColor=C_MUTED, leading=9)))
    story.append(PageBreak())
    story.append(Paragraph("ANALYSE GRAPHIQUE", s_title))
    story.append(Paragraph(f"{label}  .  Barriere {params['barriere_protect']*100:.0f}%  .  Coupon {params['coupon_annuel']*100:.1f}%/an  .  Maturite {T} ans", s_sub))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORD, spaceAfter=10))
    img_w = (W - 36*mm - 8*mm) / 2
    ih    = img_w * 0.58
    def to_img(buf, w, h):
        buf.seek(0)
        return RLImage(buf, width=w, height=h)
    g_tbl = Table([
        [to_img(buf_kde, img_w, ih),     Spacer(8*mm,1), to_img(buf_sensitivity, img_w, ih)],
        [Spacer(1,6*mm),                 Spacer(1,1),    Spacer(1,6*mm)],
        [to_img(buf_recalls, img_w, ih), Spacer(8*mm,1), to_img(buf_cumul, img_w, ih)],
    ], colWidths=[img_w, 8*mm, img_w])
    g_tbl.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"TOP"), ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("TOPPADDING",(0,0),(-1,-1),0), ("BOTTOMPADDING",(0,0),(-1,-1),0),
        ("LEFTPADDING",(0,0),(-1,-1),0), ("RIGHTPADDING",(0,0),(-1,-1),0),
    ]))
    story.append(g_tbl)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORD))
    story.append(Paragraph("Document genere par Structured Products Risk Analysis — Modele de Heston calibre sur donnees historiques. Ce document est fourni a titre informatif et ne constitue pas un conseil en investissement.", S("disc2", fontSize=6.5, textColor=C_MUTED, leading=9)))
    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(C_DARK)
        canvas.rect(0, 0, W, H, fill=1, stroke=0)
        canvas.restoreState()
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    pdf_buf.seek(0)
    return pdf_buf

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("## Parametres")
    st.markdown("---")
    st.markdown("**Sous-jacent**")
    mode = st.radio("Mode", ["Indice unique", "Panier personnalise"], label_visibility="collapsed")

    if mode == "Indice unique":
        region_A = st.selectbox("Region", list(INDICES_CATALOGUE.keys()), key="reg_a")
        choix_A  = st.selectbox("Indice", list(INDICES_CATALOGUE[region_A].keys()), key="idx_a")
        ticker_A = INDICES_CATALOGUE[region_A][choix_A]
        label_A  = choix_A
        basket_tickers_A = None
        basket_weights_A = None
    else:
        st.markdown("**Construis ton panier :**")
        st.caption("Coche des actions predefinies ou tape un ticker Yahoo Finance")
        selected_actions = {}
        for region, actions in ACTIONS_CATALOGUE.items():
            with st.expander(region, expanded=(region == "France")):
                for nom, tkr in actions.items():
                    if st.checkbox(nom, key=f"chk_{tkr}"):
                        selected_actions[nom] = tkr
        st.markdown("**Ajouter un ticker manuellement :**")
        col_tk, col_add = st.columns([3, 1])
        with col_tk:
            custom_ticker = st.text_input("Ticker", placeholder="ex: TSMC, 005930.KS...", label_visibility="collapsed")
        with col_add:
            add_ticker = st.button("+ Ajouter", use_container_width=True)
        if "custom_tickers" not in st.session_state:
            st.session_state.custom_tickers = {}
        if add_ticker and custom_ticker.strip():
            tk = custom_ticker.strip().upper()
            st.session_state.custom_tickers[tk] = tk
        for tk in list(st.session_state.custom_tickers.keys()):
            col_n, col_del = st.columns([4, 1])
            with col_n:
                st.markdown(f"<small>✓ {tk}</small>", unsafe_allow_html=True)
            with col_del:
                if st.button("x", key=f"del_{tk}"):
                    del st.session_state.custom_tickers[tk]
        all_selected = {**selected_actions, **st.session_state.custom_tickers}
        if len(all_selected) >= 2:
            st.markdown("**Poids (%)**")
            n  = len(all_selected)
            dw = round(100 / n)
            poids_raw = {}
            cols_w = st.columns(min(n, 3))
            for i, (nom, tkr) in enumerate(all_selected.items()):
                with cols_w[i % 3]:
                    poids_raw[nom] = st.number_input(nom[:7], min_value=1, max_value=100, value=dw, step=1, key=f"w_{tkr}")
            total = sum(poids_raw.values())
            if total != 100:
                st.warning(f"Total : {total}% (objectif : 100%)")
            basket_tickers_A = list(all_selected.values())
            basket_weights_A = [v / 100 for v in poids_raw.values()]
            label_A          = " / ".join(list(all_selected.keys())[:3]) + ("..." if n > 3 else "")
            ticker_A         = None
        elif len(all_selected) == 1:
            st.warning("Selectionnez au moins 2 actions.")
            basket_tickers_A = None
        else:
            st.info("Cochez des actions ou tapez un ticker.")
            basket_tickers_A = None

    st.markdown("---")
    do_comparison = st.checkbox("Comparer avec un second sous-jacent")
    if do_comparison:
        st.markdown("**Second sous-jacent**")
        mode_B = st.radio("Mode B", ["Indice unique", "Panier personnalise"], key="mb", label_visibility="collapsed")
        if mode_B == "Indice unique":
            region_B = st.selectbox("Region B", list(INDICES_CATALOGUE.keys()), key="reg_b")
            choix_B  = st.selectbox("Indice B", list(INDICES_CATALOGUE[region_B].keys()), index=1, key="idx_b")
            ticker_B = INDICES_CATALOGUE[region_B][choix_B]
            label_B  = choix_B
            basket_tickers_B = None
        else:
            selected_B = {}
            for region, actions in ACTIONS_CATALOGUE.items():
                with st.expander(region + " (B)", expanded=False):
                    for nom, tkr in actions.items():
                        if st.checkbox(nom, key=f"chkb_{tkr}"):
                            selected_B[nom] = tkr
            if len(selected_B) >= 2:
                nb  = len(selected_B)
                dwb = round(100/nb)
                poids_b = {}
                for nom, tkr in selected_B.items():
                    poids_b[nom] = st.number_input(nom[:7]+" B", min_value=1, max_value=100, value=dwb, step=1, key=f"wb_{tkr}")
                basket_tickers_B = list(selected_B.values())
                basket_weights_B = [v/100 for v in poids_b.values()]
                label_B  = " / ".join(list(selected_B.keys())[:3])
                ticker_B = None
            else:
                basket_tickers_B = None
                label_B = "?"

    st.markdown("---")
    st.markdown("**Donnees historiques**")
    start_year = st.slider("Depuis", 2010, 2022, 2015)
    end_year   = st.slider("Jusqu'a", 2020, 2025, 2025)
    START = f"{start_year}-01-01"
    END   = f"{end_year}-01-01"

    st.markdown("---")
    st.markdown("**Produit Structuré**")
    T_prod    = st.slider("Maturite (ans)", 3, 12, 8)
    coupon    = st.slider("Coupon annuel (%)", 2, 15, 6) / 100
    b_protect = st.slider("Barriere protection (%)", 40, 80, 60) / 100
    b_rappel  = st.slider("Barriere de rappel (%)", 90, 110, 100) / 100
    ATHENA_PARAMS = {"S0":100.0, "T":T_prod, "coupon_annuel":coupon,
                     "barriere_rappel":b_rappel, "barriere_protect":b_protect, "nominal":100.0}

    st.markdown("---")
    M_SIM   = st.select_slider("Nb simulations", options=[1_000,5_000,10_000,20_000], value=10_000)
    can_run = (mode == "Indice unique") or (basket_tickers_A is not None and len(basket_tickers_A) >= 2)
    run     = st.button("LANCER LA SIMULATION", use_container_width=True, disabled=not can_run)

# ── HEADER ──
st.markdown('<p class="main-title">Structured Products Risk Analysis</p>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Structured Products Risk Simulator — Heston Stochastic Volatility</p>', unsafe_allow_html=True)

# ── SIMULATION ──
if run or "res_A" in st.session_state:
    if "res_A" in st.session_state and not run:
        res_A        = st.session_state["res_A"]
        d_A          = st.session_state["d_A"]
        S_A          = st.session_state["S_A"]
        heston_A     = st.session_state["heston_A"]
        label_A      = st.session_state["label_A"]
        ATHENA_PARAMS= st.session_state["ATHENA_PARAMS"]
        T_prod       = st.session_state["T_prod"]
        coupon       = st.session_state["coupon"]
        b_protect    = st.session_state["b_protect"]
        b_rappel     = st.session_state["b_rappel"]
        M_SIM        = st.session_state["M_SIM"]
        do_comparison = False
    STEPS = 52
    with st.spinner(f"Calibration Heston sur {label_A}..."):
        try:
            prices_A = get_index(ticker_A, start=START, end=END) if mode == "Indice unique" else get_basket(basket_tickers_A, basket_weights_A, start=START, end=END)
            var_A, lr_A = compute_realized_variance(prices_A, window=21)
            heston_A    = calibrate_heston(var_A, lr_A)
            S_A, _, _   = simulate_heston(heston_A, T=T_prod, M=M_SIM, steps_per_year=STEPS)
            res_A       = pricer_athena(S_A, params=ATHENA_PARAMS, steps_per_year=STEPS)
            d_A         = res_A["details"]
            st.session_state["res_A"]      = res_A
            st.session_state["d_A"]        = d_A
            st.session_state["S_A"]        = S_A
            st.session_state["heston_A"]   = heston_A
            st.session_state["label_A"]    = label_A
            st.session_state["ATHENA_PARAMS"] = ATHENA_PARAMS
            st.session_state["T_prod"]     = T_prod
            st.session_state["coupon"]     = coupon
            st.session_state["b_protect"]  = b_protect
            st.session_state["b_rappel"]   = b_rappel
            st.session_state["M_SIM"]      = M_SIM
        except Exception as e:
            st.error(f"Erreur : {e}")
            st.stop()

    if do_comparison:
        with st.spinner(f"Calibration Heston sur {label_B}..."):
            try:
                prices_B = get_index(ticker_B, start=START, end=END) if mode_B == "Indice unique" else get_basket(basket_tickers_B, basket_weights_B, start=START, end=END)
                var_B, lr_B = compute_realized_variance(prices_B, window=21)
                heston_B    = calibrate_heston(var_B, lr_B)
                S_B, _, _   = simulate_heston(heston_B, T=T_prod, M=M_SIM, steps_per_year=STEPS)
                res_B       = pricer_athena(S_B, params=ATHENA_PARAMS, steps_per_year=STEPS)
                d_B         = res_B["details"]
            except Exception as e:
                st.error(f"Erreur {label_B} : {e}")
                do_comparison = False

    st.markdown(f'<div class="section-header">Resultats — {label_A}</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    for col, val, lbl, cls in [
        (c1, f"{d_A['P_barriere']*100:.1f}%",      "P(barriere touchee)", "metric-red"),
        (c2, f"{d_A['P_rappel']*100:.1f}%",         "P(rappel anticipe)",  "metric-green"),
        (c3, f"{d_A['duree_vie_moyenne']:.1f} ans", "Duree de vie moy.",   "metric-blue"),
        (c4, f"{d_A['payoff_moyen']:.1f}",          "Payoff moyen",        "metric-gold"),
        (c5, f"{d_A['taux_actualise']:.2f}%",       "Taux annualise moy.", "metric-blue"),
    ]:
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-value {cls}">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Parametres Heston calibres"):
        hc = heston_A
        col1,col2,col3,col4,col5 = st.columns(5)
        col1.metric("Vol initiale",   f"{np.sqrt(hc['v0'])*100:.1f}%")
        col2.metric("Vol long terme", f"{np.sqrt(hc['theta'])*100:.1f}%")
        col3.metric("Kappa",          f"{hc['kappa']:.2f}")
        col4.metric("Xi",             f"{hc['xi']:.2f}")
        col5.metric("Rho",            f"{hc['rho']:.2f}")

    st.markdown('<div class="section-header">Analyse graphique</div>', unsafe_allow_html=True)
    buf_kde  = plot_payoff_kde(res_A["payoffs"], ATHENA_PARAMS, label_A)
    buf_sens = plot_sensitivity(S_A, ATHENA_PARAMS, label_A)
    buf_rec  = plot_recall_bars(d_A["recall_proba"], T_prod, label_A)
    buf_cum  = plot_cumul_barrier(S_A, ATHENA_PARAMS, label_A)

    tab_names = ["Distribution", "Sensibilite", "Rappels", "Barriere cumulee"]
    if do_comparison: tab_names.append("Comparaison")
    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.image(buf_kde,  use_container_width=True)
        st.caption("Zone rouge = scenarios de perte. KDE lissee sur toutes les simulations.")
    with tabs[1]:
        st.image(buf_sens, use_container_width=True)
        st.caption("Impact du niveau de barriere sur le risque de perte en capital.")
    with tabs[2]:
        st.image(buf_rec,  use_container_width=True)
        st.caption("L'annee 1 concentre la majorite des rappels si le marche est positif d'emblee.")
    with tabs[3]:
        st.image(buf_cum,  use_container_width=True)
        st.caption("Fraction cumulee des trajectoires ayant touche la barriere au moins une fois.")

    if do_comparison and len(tabs) == 5:
        with tabs[4]:
            buf_cmp = plot_comparison(res_A, S_A, label_A, res_B, S_B, label_B, ATHENA_PARAMS)
            st.image(buf_cmp, use_container_width=True)
            st.markdown(f'<div class="section-header">Resultats — {label_B}</div>', unsafe_allow_html=True)
            c1,c2,c3,c4,c5 = st.columns(5)
            for col, val, lbl, cls in [
                (c1, f"{d_B['P_barriere']*100:.1f}%",      "P(barriere touchee)", "metric-red"),
                (c2, f"{d_B['P_rappel']*100:.1f}%",         "P(rappel anticipe)",  "metric-green"),
                (c3, f"{d_B['duree_vie_moyenne']:.1f} ans", "Duree de vie moy.",   "metric-blue"),
                (c4, f"{d_B['payoff_moyen']:.1f}",          "Payoff moyen",        "metric-gold"),
                (c5, f"{d_B['taux_actualise']:.2f}%",       "Taux annualise moy.", "metric-blue"),
            ]:
                with col:
                    st.markdown(f'<div class="metric-card"><div class="metric-value {cls}">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

    v0_pct = np.sqrt(heston_A["v0"]) * 100
    th_pct = np.sqrt(heston_A["theta"]) * 100
    st.markdown('<div class="section-header">Fiche Produit</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="fiche">
        <div class="fiche-title">TERM SHEET — ATHENA</div>
        <div class="fiche-row"><span class="fiche-key">Sous-jacent</span><span class="fiche-val">{label_A}</span></div>
        <div class="fiche-row"><span class="fiche-key">Maturite</span><span class="fiche-val">{T_prod} ans</span></div>
        <div class="fiche-row"><span class="fiche-key">Coupon annuel</span><span class="fiche-val">{coupon*100:.1f}% cumulatif</span></div>
        <div class="fiche-row"><span class="fiche-key">Barriere rappel</span><span class="fiche-val">{b_rappel*100:.0f}%</span></div>
        <div class="fiche-row"><span class="fiche-key">Barriere protection</span><span class="fiche-val">{b_protect*100:.0f}%</span></div>
        <br>
        <div class="fiche-title">RESULTATS ({M_SIM:,} simulations)</div>
        <div class="fiche-row"><span class="fiche-key">P(barriere touchee)</span><span class="fiche-val" style="color:{RED}">{d_A['P_barriere']*100:.1f}%</span></div>
        <div class="fiche-row"><span class="fiche-key">P(rappel anticipe)</span><span class="fiche-val" style="color:{GREEN}">{d_A['P_rappel']*100:.1f}%</span></div>
        <div class="fiche-row"><span class="fiche-key">P(perte a maturite)</span><span class="fiche-val" style="color:{RED}">{d_A['P_perte_maturite']*100:.1f}%</span></div>
        <div class="fiche-row"><span class="fiche-key">Payoff moyen</span><span class="fiche-val">{d_A['payoff_moyen']:.1f}  (P10: {d_A['payoff_p10']:.1f} / P90: {d_A['payoff_p90']:.1f})</span></div>
        <div class="fiche-row"><span class="fiche-key">Duree de vie moyenne</span><span class="fiche-val">{d_A['duree_vie_moyenne']:.1f} ans</span></div>
        <div class="fiche-row"><span class="fiche-key">Taux annualise moyen</span><span class="fiche-val">{d_A['taux_actualise']:.2f}%</span></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
    with st.spinner("Generation du PDF..."):
        b2  = plot_payoff_kde(res_A["payoffs"], ATHENA_PARAMS, label_A)
        b2s = plot_sensitivity(S_A, ATHENA_PARAMS, label_A)
        b2r = plot_recall_bars(d_A["recall_proba"], T_prod, label_A)
        b2c = plot_cumul_barrier(S_A, ATHENA_PARAMS, label_A)
        pdf_buf = generate_pdf(label=label_A, params=ATHENA_PARAMS, heston_params=heston_A,
            details=d_A, recall_proba=d_A["recall_proba"],
            buf_kde=b2, buf_sensitivity=b2s, buf_recalls=b2r, buf_cumul=b2c)

    col_pdf, col_txt = st.columns(2)
    with col_pdf:
        st.download_button(label="Telecharger le PDF client (2 pages)", data=pdf_buf,
            file_name=f"athena_{label_A.replace(' ','_').replace('/','_')}.pdf",
            mime="application/pdf", use_container_width=True)
    with col_txt:
        fiche_txt = (
            f"Structured Products Risk Analysis — {label_A}\n{'='*50}\n"
            f"Maturite : {T_prod} ans  |  Coupon : {coupon*100:.1f}%/an\n"
            f"Barriere protection : {b_protect*100:.0f}%  |  Barriere rappel : {b_rappel*100:.0f}%\n\n"
            f"RESULTATS ({M_SIM:,} simulations)\n"
            f"P(barriere touchee)  : {d_A['P_barriere']*100:.1f}%\n"
            f"P(rappel anticipe)   : {d_A['P_rappel']*100:.1f}%\n"
            f"P(perte a maturite)  : {d_A['P_perte_maturite']*100:.1f}%\n"
            f"Payoff moyen         : {d_A['payoff_moyen']:.1f}\n"
            f"Payoff P10 / P90     : {d_A['payoff_p10']:.1f} / {d_A['payoff_p90']:.1f}\n"
            f"Duree de vie moy.    : {d_A['duree_vie_moyenne']:.1f} ans\n"
            f"Taux annualise moy.  : {d_A['taux_actualise']:.2f}%\n\n"
            f"HESTON CALIBRE\n"
            f"v0={v0_pct:.1f}%  theta={th_pct:.1f}%  kappa={heston_A['kappa']}  xi={heston_A['xi']}  rho={heston_A['rho']}\n"
        )
        st.download_button(label="Telecharger la fiche (.txt)", data=fiche_txt,
            file_name=f"athena_{label_A.replace(' ','_').replace('/','_')}.txt",
            mime="text/plain", use_container_width=True)

else:
    st.markdown("""
    <div style="margin-top:6rem; text-align:center; color:#334155;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:4rem;">◈</div>
        <div style="font-size:1rem; margin-top:1rem; color:#475569;">
            Configurez le sous-jacent et les parametres dans la barre laterale,<br>
            puis lancez la simulation.
        </div>
        <div style="font-size:0.75rem; margin-top:2rem; color:#334155; font-family:'IBM Plex Mono',monospace;">
            Modele · Heston (vol stochastique) calibre sur donnees reelles<br>
            Methode · Monte Carlo — calibration AR(1) sur variance realisee
        </div>
    </div>
    """, unsafe_allow_html=True)
