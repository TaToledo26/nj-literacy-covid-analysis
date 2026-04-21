"""
03_visualize.py
---------------
NJ Literacy & COVID Economic Impact — Visualization Script

Produces two figures saved to output/figures/:

  fig1_ela_by_dfg_pre_post_covid.png
      Grouped bar chart showing mean ELA proficiency by DFG socioeconomic
      tier, comparing the Pre-COVID baseline (2017-2019) against the
      Post-COVID Recovery period (2021-2024). Percentage labels appear on
      every bar; the change in percentage points is annotated below each pair.
      The NJ 2030 goal of 80% proficiency is marked with a reference line.

  fig2_gap_widening.png
      Two-panel chart. Top panel: trend lines for all 8 DFG tiers across
      the five assessment years, with the gap between the highest and lowest
      tiers shaded. Bottom panel: bar chart showing the raw proficiency gap
      (DFG-J minus DFG-A) in each year, illustrating that the gap has grown
      since COVID.

Usage:
  Must be run AFTER 01_clean_data.py (reads data/processed/ela_by_dfg_year.csv)
  python scripts/03_visualize.py

Requirements:
  pip install pandas matplotlib numpy
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────────────────
PROC_DIR   = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("output", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
DFG_ORDER = ["A", "B", "CD", "DE", "FG", "GH", "I", "J"]

DFG_LABELS = {
    "A":  "A\n(Lowest)",
    "B":  "B",
    "CD": "CD",
    "DE": "DE",
    "FG": "FG",
    "GH": "GH",
    "I":  "I",
    "J":  "J\n(Highest)",
}

# Color per DFG tier: red = low SES → blue = high SES
TIER_PALETTE = [
    "#E03030",  # A
    "#E85D26",  # B
    "#D4841A",  # CD
    "#C8A820",  # DE
    "#6DB56D",  # FG
    "#3AB5B5",  # GH
    "#3A7FD4",  # I
    "#5B4FE8",  # J
]

# Assessment years only (COVID years had no NJSLA)
ASSESSMENT_YEARS  = ["2017-2018", "2018-2019", "2021-2022", "2022-2023", "2023-2024"]
ASSESSMENT_LABELS = ["2017-18",   "2018-19",   "2021-22",   "2022-23",   "2023-24"]


# ════════════════════════════════════════════════════════════════════════════
# Helper
# ════════════════════════════════════════════════════════════════════════════

def dark_axes(ax):
    """Apply dark-theme styling to a matplotlib Axes object."""
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444")
    ax.spines["bottom"].set_color("#444")
    ax.tick_params(colors="#AAAAAA", length=0)
    ax.yaxis.grid(True, color="#2a2a3a", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Grouped bar chart: Pre-COVID vs Recovery by DFG tier
# ════════════════════════════════════════════════════════════════════════════

def make_fig1(dfg: pd.DataFrame) -> None:
    """
    Grouped bar chart comparing mean ELA proficiency before and after COVID
    for each DFG socioeconomic tier. Bars are labeled with percentages; the
    change (in pp) is shown below each group.
    """
    # Aggregate to one value per tier per period
    pre  = (dfg[dfg["period"] == "Pre-COVID"]
            .groupby("dfg")["ela_proficiency_mean"].mean()
            .reindex(DFG_ORDER))
    post = (dfg[dfg["period"] == "Recovery"]
            .groupby("dfg")["ela_proficiency_mean"].mean()
            .reindex(DFG_ORDER))
    change = post - pre

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("#FFFFFF")
    dark_axes(ax)

    x     = np.arange(len(DFG_ORDER))
    width = 0.35

    COLOR_PRE  = "#4A90D9"   # steel blue  — Pre-COVID bars
    COLOR_POST = "#E8754A"   # burnt orange — Recovery bars

    bars_pre  = ax.bar(x - width / 2, pre.values,  width,
                       label="Pre-COVID (2017–2019)",
                       color=COLOR_PRE,  alpha=1, zorder=3)
    bars_post = ax.bar(x + width / 2, post.values, width,
                       label="Recovery (2021–2024)",
                       color=COLOR_POST, alpha=1, zorder=3)

    # ── Percentage labels on every bar ───────────────────────────────────
    for bar in bars_pre:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                    f"{h:.1f}%", ha="center", va="bottom",
                    fontsize=8.5, color="#AEC6E8", fontweight="bold")

    for bar in bars_post:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                    f"{h:.1f}%", ha="center", va="bottom",
                    fontsize=8.5, color="#F5AA88", fontweight="bold")

    # ── Change annotations below each pair ───────────────────────────────
    for i, tier in enumerate(DFG_ORDER):
        ch = change[tier]
        if not np.isnan(ch):
            ax.text(i, -6.5, f"{ch:+.1f}pp",
                    ha="center", va="center",
                    fontsize=9, color="#FF6B6B", fontweight="bold")

    ax.text(0.5, -0.11,
            "Change from Pre-COVID baseline (percentage points)",
            transform=ax.transAxes, ha="center", fontsize=9,
            color="#FF6B6B", style="italic")

    # ── NJ 2030 goal line ────────────────────────────────────────────────
    ax.axhline(80, color="#FFD700", linewidth=1.3, linestyle="--", zorder=4)
    ax.text(len(DFG_ORDER) - 0.42, 80.8, "NJ 2030 Goal: 80%",
            color="#FFD700", fontsize=8.5, ha="right")

    # ── Axis formatting ───────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels([DFG_LABELS[t] for t in DFG_ORDER],
                       color="white", fontsize=10)
    ax.set_yticks(range(0, 101, 10))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 10)],
                       color="#AAAAAA", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.6, len(DFG_ORDER) - 0.4)

    ax.set_xlabel("District Factor Group (DFG) — Socioeconomic Tier",
                  color="white", fontsize=11, labelpad=20)
    ax.set_ylabel("Mean ELA Proficiency (%)", color="white", fontsize=11, labelpad=10)
    ax.set_title(
        "NJ ELA Proficiency by Economic Tier\nPre-COVID vs. Post-COVID Recovery",
        color="white", fontsize=14, fontweight="bold", pad=16
    )
    ax.legend(fontsize=10, framealpha=0.2, facecolor="#1a1a2e",
              edgecolor="#444", labelcolor="black", loc="upper left")

    fig.text(
        0.5, 0.01,
        "Source: NJ DOE School Performance Reports 2017–2024  |"
        "  DFG A = lowest income, J = highest income",
        ha="center", fontsize=8, color="#666", style="italic"
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(OUTPUT_DIR, "fig1_ela_by_dfg_pre_post_covid.png")
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Gap widening: trend lines + gap bar chart
# ════════════════════════════════════════════════════════════════════════════

def make_fig2(dfg: pd.DataFrame) -> None:
    """
    Two-panel chart showing the widening literacy gap across DFG tiers.

    Top panel: one trend line per DFG tier for all 5 assessment years,
               with DFG-A and DFG-J highlighted and the gap between them
               shaded.
    Bottom panel: bar chart of the raw gap (DFG-J minus DFG-A) per year,
                  confirming the gap has grown since COVID.
    """
    x_pos = np.arange(len(ASSESSMENT_YEARS))

    # Extract tier-A and tier-J series for gap calculation
    tier_a, tier_j = [], []
    for yr in ASSESSMENT_YEARS:
        sub = dfg[dfg["school_year"] == yr]
        a_v = sub[sub["dfg"] == "A"]["ela_proficiency_mean"].values
        j_v = sub[sub["dfg"] == "J"]["ela_proficiency_mean"].values
        tier_a.append(a_v[0] if len(a_v) else np.nan)
        tier_j.append(j_v[0] if len(j_v) else np.nan)

    tier_a = np.array(tier_a)
    tier_j = np.array(tier_j)
    gap    = tier_j - tier_a

    fig, axes = plt.subplots(2, 1, figsize=(12, 10),
                             gridspec_kw={"height_ratios": [2.2, 1]})
    fig.patch.set_facecolor("#FFFFFF")

    # ── TOP PANEL: all 8 tier trend lines ────────────────────────────────
    ax1 = axes[0]
    dark_axes(ax1)

    for i, tier in enumerate(DFG_ORDER):
        vals = []
        for yr in ASSESSMENT_YEARS:
            sub = dfg[dfg["school_year"] == yr]
            v = sub[sub["dfg"] == tier]["ela_proficiency_mean"].values
            vals.append(v[0] if len(v) else np.nan)
        vals  = np.array(vals)
        emph  = tier in ("A", "J")
        alpha = 1.0 if emph else 0.55
        lw    = 2.5 if emph else 1.2
        ms    = 5   if emph else 3

        ax1.plot(x_pos, vals, color=TIER_PALETTE[i], lw=lw, alpha=alpha,
                 marker="o", markersize=ms, label=f"DFG {tier}", zorder=3)

        # Right-side endpoint label
        if not np.isnan(vals[-1]):
            ax1.text(x_pos[-1] + 0.07, vals[-1],
                     f"{tier}: {vals[-1]:.0f}%",
                     color=TIER_PALETTE[i], fontsize=8, va="center",
                     fontweight="bold" if emph else "normal")

    # Shade gap between A and J
    ax1.fill_between(x_pos, tier_a, tier_j,
                     alpha=0.08, color="#FFFFFF", zorder=1)

    # COVID gap annotation band
    ax1.axvspan(1.5, 2.5, color="#FF6B6B", alpha=0.07, zorder=0)
    ax1.text(2.0, 5, "COVID\ngap", ha="center", fontsize=8,
             color="#FF6B6B", alpha=0.8, style="italic")

    # NJ 2030 goal line
    ax1.axhline(80, color="#7292c2", lw=1.2, ls="--", zorder=4)
    ax1.text(x_pos[-1] + 0.07, 80.5, "80% goal",
             color="#7292c2", fontsize=7.5)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([])
    ax1.set_yticks(range(0, 101, 10))
    ax1.set_yticklabels([f"{v}%" for v in range(0, 101, 10)],
                        color="#AAAAAA", fontsize=9)
    ax1.set_ylim(0, 100)
    ax1.set_xlim(-0.3, len(x_pos) - 0.3)
    ax1.set_ylabel("Mean ELA Proficiency (%)", color="black", fontsize=10)
    ax1.set_title(
        "The Literacy Gap is Growing — and Not Closing\n"
        "ELA Proficiency by DFG Tier Before and After COVID-19",
        color="black", fontsize=13, fontweight="bold", pad=12
    )
    ax1.legend(ncol=4, fontsize=8, framealpha=0.15, facecolor="#1a1a2e",
               edgecolor="#444", labelcolor="black", loc="lower left")

    # ── BOTTOM PANEL: gap bar chart ───────────────────────────────────────
    ax2 = axes[1]
    dark_axes(ax2)

    max_gap    = np.nanmax(gap)
    bar_colors = ["#E03030" if g > max_gap * 0.85 else "#D4841A" for g in gap]
    bars = ax2.bar(x_pos, gap, color=bar_colors, alpha=0.9, zorder=3, width=0.55)

    for bar, g in zip(bars, gap):
        if not np.isnan(g):
            ax2.text(bar.get_x() + bar.get_width() / 2, g + 0.4,
                     f"{g:.1f}pp", ha="center", va="bottom",
                     fontsize=10, color="white", fontweight="bold")

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ASSESSMENT_LABELS, color="white", fontsize=10)
    ax2.set_ylabel("Proficiency Gap\n(DFG-J minus DFG-A)",
                   color="#AAAAAA", fontsize=9)
    ax2.set_ylim(0, max_gap * 1.25)
    ax2.set_xlim(-0.5, len(x_pos) - 0.5)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v)}pp")
    )
    ax2.tick_params(axis="y", labelcolor="#AAAAAA", labelsize=8)

    fig.text(
        0.5, 0.01,
        "Source: NJ DOE School Performance Reports 2017–2024  |  "
        "COVID years (2019-20, 2020-21) excluded — no NJSLA assessment conducted",
        ha="center", fontsize=7.5, color="#555", style="italic"
    )

    plt.tight_layout(rect=[0, 0.03, 0.93, 1])
    path = os.path.join(OUTPUT_DIR, "fig2_gap_widening.png")
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("NJ Literacy & COVID Economic Impact — 03_visualize.py")
    print("=" * 62)

    dfg_path = os.path.join(PROC_DIR, "ela_by_dfg_year.csv")
    if not os.path.exists(dfg_path):
        print(f"\n❌ File not found: {dfg_path}")
        print("   Run 01_clean_data.py first.")
        return

    dfg = pd.read_csv(dfg_path)
    print(f"\n  Loaded {len(dfg)} DFG-tier × school-year rows\n")

    print("  Generating Figure 1 — Pre/Post COVID bar chart...")
    make_fig1(dfg)

    print("  Generating Figure 2 — Gap widening chart...")
    make_fig2(dfg)

    print("\n" + "=" * 62)
    print("VISUALIZATION COMPLETE")
    print(f"Figures saved to: {OUTPUT_DIR}/")
    print("=" * 62)


if __name__ == "__main__":
    main()
    

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
PROC_DIR   = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("output", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DFG_CSV  = os.path.join(PROC_DIR, "ela_by_dfg_year.csv")
DIST_CSV = os.path.join(PROC_DIR, "ela_by_district_year.csv")
REG_CSV  = os.path.join(PROC_DIR, "regression_dataset.csv")
SUM_CSV  = os.path.join(PROC_DIR, "dfg_recovery_summary.csv")

# ── Constants ──────────────────────────────────────────────────────────────
DFG_ORDER  = ["A", "B", "CD", "DE", "FG", "GH", "I", "J"]
DFG_LABELS = {"A": "A\n(Lowest)", "B": "B", "CD": "CD", "DE": "DE",
              "FG": "FG", "GH": "GH", "I": "I",  "J": "J\n(Highest)"}
NJ_GOAL    = 80.0

ASSESSMENT_YEARS  = ["2017-2018", "2018-2019", "2021-2022", "2022-2023", "2023-2024"]
ASSESSMENT_LABELS = ["2017-18",   "2018-19",   "2021-22",   "2022-23",   "2023-24"]

# Color per DFG tier: red (low SES) → blue (high SES)
TIER_PALETTE = ["#E03030", "#E85D26", "#D4841A", "#C8A820",
                "#6DB56D", "#3AB5B5", "#3A7FD4", "#5B4FE8"]
TIER_COLOR   = dict(zip(DFG_ORDER, TIER_PALETTE))

BG        = "#0f1117"
BG_PANEL  = "#1a1a2e"
GRID_COL  = "#2a2a3a"
SPINE_COL = "#444444"
TEXT_COL  = "white"
MUTED_COL = "#AAAAAA"
GOLD      = "#FFD700"
SOURCE    = ("Source: NJ DOE School Performance Reports 2017–2024  |  "
             "DFG A = lowest income, J = highest income")


def dark_ax(ax):
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COL)
    ax.spines["bottom"].set_color(SPINE_COL)
    ax.tick_params(colors=MUTED_COL, length=0)
    ax.yaxis.grid(True, color=GRID_COL, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
# FIG 1 — Grouped bar chart: Pre-COVID vs Recovery by DFG tier
# ════════════════════════════════════════════════════════════════════════════

def make_fig1(dfg):
    pre    = (dfg[dfg["period"] == "Pre-COVID"]
              .groupby("dfg")["ela_proficiency_mean"].mean().reindex(DFG_ORDER))
    post   = (dfg[dfg["period"] == "Recovery"]
              .groupby("dfg")["ela_proficiency_mean"].mean().reindex(DFG_ORDER))
    change = post - pre

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(BG)
    dark_ax(ax)

    x, w = np.arange(len(DFG_ORDER)), 0.35
    COLOR_PRE  = "#4A90D9"
    COLOR_POST = "#E8754A"

    b_pre  = ax.bar(x - w/2, pre.values,  w, label="Pre-COVID (2017–2019)",
                    color=COLOR_PRE,  alpha=0.92, zorder=3)
    b_post = ax.bar(x + w/2, post.values, w, label="Recovery (2021–2024)",
                    color=COLOR_POST, alpha=0.92, zorder=3)

    for bar, color in [(b_pre, "#AEC6E8"), (b_post, "#F5AA88")]:
        for b in bar:
            h = b.get_height()
            if not np.isnan(h):
                ax.text(b.get_x() + b.get_width()/2, h + 0.8,
                        f"{h:.1f}%", ha="center", va="bottom",
                        fontsize=8.5, color=color, fontweight="bold")

    for i, tier in enumerate(DFG_ORDER):
        ch = change[tier]
        if not np.isnan(ch):
            ax.text(i, -6.5, f"{ch:+.1f}pp", ha="center", va="center",
                    fontsize=9, color="#FF6B6B", fontweight="bold")

    ax.axhline(NJ_GOAL, color=GOLD, linewidth=1.3, linestyle="--", zorder=4)
    ax.text(len(DFG_ORDER) - 0.42, NJ_GOAL + 0.8, "NJ 2030 Goal: 80%",
            color=GOLD, fontsize=8.5, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([DFG_LABELS[t] for t in DFG_ORDER], color=TEXT_COL, fontsize=10)
    ax.set_yticks(range(0, 101, 10))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 10)], color=MUTED_COL, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.6, len(DFG_ORDER) - 0.4)
    ax.set_xlabel("District Factor Group (DFG) — Socioeconomic Tier",
                  color=TEXT_COL, fontsize=11, labelpad=20)
    ax.set_ylabel("Mean ELA Proficiency (%)", color=TEXT_COL, fontsize=11, labelpad=10)
    ax.set_title("NJ ELA Proficiency by Economic Tier\nPre-COVID vs. Post-COVID Recovery",
                 color=TEXT_COL, fontsize=14, fontweight="bold", pad=16)
    ax.legend(fontsize=10, framealpha=0.2, facecolor=BG_PANEL,
              edgecolor=SPINE_COL, labelcolor=TEXT_COL, loc="upper left")
    fig.text(0.5, 0.01, SOURCE, ha="center", fontsize=8, color="#666", style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, "fig1_ela_by_dfg_pre_post_covid.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 2 — Gap widening: trend lines + gap bar
# ════════════════════════════════════════════════════════════════════════════

def make_fig2(dfg):
    x_pos  = np.arange(len(ASSESSMENT_YEARS))
    tier_a = np.array([
        dfg[(dfg["school_year"] == yr) & (dfg["dfg"] == "A")]["ela_proficiency_mean"].values
        for yr in ASSESSMENT_YEARS
    ], dtype=object)
    tier_j = np.array([
        dfg[(dfg["school_year"] == yr) & (dfg["dfg"] == "J")]["ela_proficiency_mean"].values
        for yr in ASSESSMENT_YEARS
    ], dtype=object)

    tier_a = np.array([v[0] if len(v) else np.nan for v in tier_a], dtype=float)
    tier_j = np.array([v[0] if len(v) else np.nan for v in tier_j], dtype=float)
    gap    = tier_j - tier_a

    fig, axes = plt.subplots(2, 1, figsize=(12, 10),
                             gridspec_kw={"height_ratios": [2.2, 1]})
    fig.patch.set_facecolor(BG)

    ax1 = axes[0]
    dark_ax(ax1)

    for i, tier in enumerate(DFG_ORDER):
        vals = np.array([
            (lambda v: v[0] if len(v) else np.nan)(
                dfg[(dfg["school_year"] == yr) & (dfg["dfg"] == tier)]["ela_proficiency_mean"].values
            )
            for yr in ASSESSMENT_YEARS
        ], dtype=float)
        emph  = tier in ("A", "J")
        ax1.plot(x_pos, vals, color=TIER_PALETTE[i],
                 lw=2.5 if emph else 1.2, alpha=1.0 if emph else 0.55,
                 marker="o", markersize=5 if emph else 3,
                 label=f"DFG {tier}", zorder=3)
        if not np.isnan(vals[-1]):
            ax1.text(x_pos[-1] + 0.07, vals[-1], f"{tier}: {vals[-1]:.0f}%",
                     color=TIER_PALETTE[i], fontsize=8, va="center",
                     fontweight="bold" if emph else "normal")

    ax1.fill_between(x_pos, tier_a, tier_j, alpha=0.08, color=TEXT_COL, zorder=1)
    ax1.axvspan(1.5, 2.5, color="#FF6B6B", alpha=0.07, zorder=0)
    ax1.text(2.0, 5, "COVID\ngap", ha="center", fontsize=8,
             color="#FF6B6B", alpha=0.8, style="italic")
    ax1.axhline(NJ_GOAL, color=GOLD, lw=1.2, ls="--", zorder=4)
    ax1.text(x_pos[-1] + 0.07, NJ_GOAL + 0.5, "80% goal", color=GOLD, fontsize=7.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([])
    ax1.set_yticks(range(0, 101, 10))
    ax1.set_yticklabels([f"{v}%" for v in range(0, 101, 10)], color=MUTED_COL, fontsize=9)
    ax1.set_ylim(0, 100)
    ax1.set_xlim(-0.3, len(x_pos) - 0.3)
    ax1.set_ylabel("Mean ELA Proficiency (%)", color=TEXT_COL, fontsize=10)
    ax1.set_title("The Literacy Gap is Growing — and Not Closing\n"
                  "ELA Proficiency by DFG Tier Before and After COVID-19",
                  color=TEXT_COL, fontsize=13, fontweight="bold", pad=12)
    ax1.legend(ncol=4, fontsize=8, framealpha=0.15, facecolor=BG_PANEL,
               edgecolor=SPINE_COL, labelcolor=TEXT_COL, loc="lower left")

    ax2 = axes[1]
    dark_ax(ax2)
    max_gap    = np.nanmax(gap)
    bar_colors = ["#E03030" if g > max_gap * 0.85 else "#D4841A" for g in gap]
    bars = ax2.bar(x_pos, gap, color=bar_colors, alpha=0.9, zorder=3, width=0.55)
    for bar, g in zip(bars, gap):
        if not np.isnan(g):
            ax2.text(bar.get_x() + bar.get_width()/2, g + 0.4,
                     f"{g:.1f}pp", ha="center", va="bottom",
                     fontsize=10, color=TEXT_COL, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ASSESSMENT_LABELS, color=TEXT_COL, fontsize=10)
    ax2.set_ylabel("Gap: DFG-J minus DFG-A", color=MUTED_COL, fontsize=9)
    ax2.set_ylim(0, max_gap * 1.25)
    ax2.set_xlim(-0.5, len(x_pos) - 0.5)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}pp"))
    ax2.tick_params(axis="y", labelcolor=MUTED_COL, labelsize=8)

    fig.text(0.5, 0.01,
             f"{SOURCE}  |  COVID years (2019-20, 2020-21) excluded — no NJSLA conducted",
             ha="center", fontsize=7.5, color="#555", style="italic")
    plt.tight_layout(rect=[0, 0.03, 0.93, 1])
    save(fig, "fig2_gap_widening.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 6 — Regression scatter: econ disadvantaged vs recovery change
# ════════════════════════════════════════════════════════════════════════════

def make_fig3(reg):
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(BG)
    dark_ax(ax)

    # Scatter points colored by DFG tier
    for i, tier in enumerate(DFG_ORDER):
        sub = reg[reg["dfg"] == tier]
        if sub.empty:
            continue
        ax.scatter(sub["econ_disadvantaged"], sub["recovery_change"],
                   color=TIER_PALETTE[i], alpha=0.65, s=30, zorder=3,
                   label=f"DFG {tier}", edgecolors="none")

    # OLS regression line with 95% CI band
    x_vals = reg["econ_disadvantaged"].values
    y_vals = reg["recovery_change"].values
    slope, intercept, r, p, se = stats.linregress(x_vals, y_vals)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
    y_line = slope * x_line + intercept

    # 95% CI
    n    = len(x_vals)
    t_cv = stats.t.ppf(0.975, df=n - 2)
    x_mean = x_vals.mean()
    se_line = se * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((x_vals - x_mean)**2))
    ci = t_cv * se_line

    ax.plot(x_line, y_line, color="#FF6B6B", lw=2, zorder=4, label="OLS Regression Line")
    ax.fill_between(x_line, y_line - ci, y_line + ci,
                    color="#FF6B6B", alpha=0.12, zorder=2, label="95% Confidence Band")

    # Zero line
    ax.axhline(0, color=SPINE_COL, lw=1, ls="--", zorder=1)

    # Annotation box with regression stats
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    stats_text = (
        f"β = {slope:.3f}  SE = {se:.3f}\n"
        f"r = {r:.3f}    p = {p:.4f} {sig}\n"
        f"n = {n} districts"
    )
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=TEXT_COL,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_PANEL,
                      edgecolor=SPINE_COL, alpha=0.85))

    ax.set_xlabel("% Economically Disadvantaged Students (2017-2018)",
                  color=TEXT_COL, fontsize=11, labelpad=8)
    ax.set_ylabel("ELA Recovery Change 2017-18 → 2023-24 (pp)",
                  color=TEXT_COL, fontsize=11, labelpad=8)
    ax.set_title("Economic Disadvantage Predicts Slower Literacy Recovery\n"
                 "District-Level OLS: % Econ Disadvantaged vs. ELA Recovery Change",
                 color=TEXT_COL, fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelcolor=MUTED_COL, labelsize=9)
    ax.tick_params(axis="y", labelcolor=MUTED_COL, labelsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}pp"))

    legend = ax.legend(ncol=2, fontsize=8.5, framealpha=0.2, facecolor=BG_PANEL,
                       edgecolor=SPINE_COL, labelcolor=TEXT_COL, loc="lower right")

    fig.text(0.5, 0.01, SOURCE, ha="center", fontsize=8, color="#666", style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, "fig6_regression_scatter.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 7 — 2030 Goal Projection: horizontal bar chart
# ════════════════════════════════════════════════════════════════════════════

def make_fig4(summary):
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(BG)
    dark_ax(ax)
    ax.xaxis.grid(True, color=GRID_COL, linewidth=0.7, zorder=0)
    ax.yaxis.grid(False)

    tiers    = DFG_ORDER[::-1]   # plot A at bottom, J at top
    y_pos    = np.arange(len(tiers))
    bar_h    = 0.32

    for i, tier in enumerate(tiers):
        row  = summary[summary["dfg"] == tier].iloc[0]
        col  = TIER_COLOR[tier]
        y    = y_pos[i]

        # 2024 actual (solid bar)
        ax.barh(y + bar_h/2, row["ela_post_mean"], bar_h,
                color=col, alpha=0.9, zorder=3)

        # 2030 projected (hatched, lighter)
        proj = row["projected_2030"]
        if not pd.isna(proj):
            proj_col = "#4CAF50" if row["on_track_2030"] else "#E85D26"
            ax.barh(y - bar_h/2, proj, bar_h,
                    color=proj_col, alpha=0.55, hatch="///",
                    edgecolor=proj_col, linewidth=0.4, zorder=3)

        # Labels
        ax.text(row["ela_post_mean"] + 0.5, y + bar_h/2,
                f"{row['ela_post_mean']:.1f}%",
                va="center", fontsize=8.5, color=col, fontweight="bold")
        if not pd.isna(proj):
            label_color = "#4CAF50" if row["on_track_2030"] else "#E85D26"
            ax.text(proj + 0.5, y - bar_h/2,
                    f"{proj:.1f}% ({'✓' if row['on_track_2030'] else '✗'})",
                    va="center", fontsize=8.5, color=label_color)

    # 80% goal line
    ax.axvline(NJ_GOAL, color=GOLD, lw=1.5, ls="--", zorder=4)
    ax.text(NJ_GOAL + 0.3, len(tiers) - 0.3, "80%\nGoal",
            color=GOLD, fontsize=8.5, va="top")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"DFG {t}  ({['Lowest','','','','','','','Highest'][DFG_ORDER.index(t)]})" if t in ('A','J')
         else f"DFG {t}" for t in tiers],
        color=TEXT_COL, fontsize=10
    )
    ax.set_xlabel("Mean ELA Proficiency (%)", color=TEXT_COL, fontsize=11, labelpad=8)
    ax.set_xlim(0, 105)
    ax.tick_params(axis="x", labelcolor=MUTED_COL, labelsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))

    ax.set_title("Distance from NJ 2030 Literacy Goal by Economic Tier\n"
                 "2023–2024 Actual vs. Projected 2030 at Current Recovery Rate",
                 color=TEXT_COL, fontsize=13, fontweight="bold", pad=12)

    # Legend
    solid_patch = mpatches.Patch(color="#888", label="2023-24 Actual")
    green_patch = mpatches.Patch(color="#4CAF50", alpha=0.6, hatch="///",
                                 label="Projected 2030 (on track ✓)")
    red_patch   = mpatches.Patch(color="#E85D26", alpha=0.6, hatch="///",
                                 label="Projected 2030 (off track ✗)")
    ax.legend(handles=[solid_patch, green_patch, red_patch],
              fontsize=9, framealpha=0.2, facecolor=BG_PANEL,
              edgecolor=SPINE_COL, labelcolor=TEXT_COL, loc="lower right")

    fig.text(0.5, 0.01, SOURCE, ha="center", fontsize=8, color="#666", style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, "fig7_goal_projection.png")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  NJ Literacy & COVID Economic Impact — 03_visualize.py")
    print("=" * 62)

    missing = [p for p in [DFG_CSV, DIST_CSV, REG_CSV, SUM_CSV]
               if not os.path.exists(p)]
    if missing:
        print("\n  Missing required files:")
        for p in missing:
            print(f"    {p}")
        print("\n  Run 01_clean_data.py then 02_analysis.py first.")
        return

    dfg     = pd.read_csv(DFG_CSV)
    reg     = pd.read_csv(REG_CSV)
    summary = pd.read_csv(SUM_CSV)

    print(f"\n  DFG rows      : {len(dfg):,}")
    print(f"  Reg districts : {len(reg):,}")
    print(f"  DFG tiers     : {len(summary)}")

    print("\n  Generating Figure 1 — Pre/Post COVID bar chart...")
    make_fig1(dfg)

    print("  Generating Figure 2 — Gap widening trend chart...")
    make_fig2(dfg)

    print("  Generating Figure 3 — Regression scatter plot...")
    make_fig3(reg)

    print("  Generating Figure 4 — 2030 goal projection...")
    make_fig4(summary)

    print("\n" + "=" * 62)
    print("  VISUALIZATION COMPLETE")
    print(f"  All figures saved to: {OUTPUT_DIR}/")
    print("=" * 62)


if __name__ == "__main__":
    main()