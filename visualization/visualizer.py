# visualization/visualizer.py
#
# All visualization logic lives here.
# Generates three outputs:
#   1. Susceptibility Heatmap — departments vs Cialdini principles
#   2. Vulnerability Network Graph — employee nodes colored by risk
#   3. Individual Risk Radar Chart — OCEAN + susceptibility per person

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
import seaborn as sns

from config import (
    CIALDINI_PRINCIPLES,
    DEPARTMENTS,
    OUTPUT_DIR,
    GRAPH_OUTPUT_FILE,
    HEATMAP_OUTPUT_FILE,
    REPORT_OUTPUT_FILE,
    OCEAN_TRAITS,
)

# ------------------------------------------------------------------
# Shared style settings
# ------------------------------------------------------------------
RISK_COLORS = {
    "HIGH":   "#E74C3C",   # Red
    "MEDIUM": "#F39C12",   # Orange
    "LOW":    "#2ECC71",   # Green
}

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------------
# 1. Department Susceptibility Heatmap
# ------------------------------------------------------------------

def plot_heatmap(dept_averages: dict, save_path: str = HEATMAP_OUTPUT_FILE):
    """
    Plots a heatmap of average susceptibility scores across
    departments (rows) and Cialdini principles (columns).

    Red = highly susceptible. Green = low susceptibility.

    Args:
        dept_averages: Output of aggregate_department_risk()
        save_path: File path to save the figure.
    """
    _ensure_output_dir()

    # Build DataFrame: rows = departments, columns = principles
    dept_list = [d for d in DEPARTMENTS if d in dept_averages]
    data = {
        dept: [dept_averages[dept].get(p, 0.0) for p in CIALDINI_PRINCIPLES]
        for dept in dept_list
    }
    df = pd.DataFrame(data, index=CIALDINI_PRINCIPLES).T

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(
        df,
        ax=ax,
        cmap="RdYlGn_r",   # Red = high risk, Green = low risk
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="#CCCCCC",
        cbar_kws={"label": "Susceptibility Score (0 = Low, 1 = High)"},
    )

    ax.set_title(
        "Department-Level Social Engineering Susceptibility Map\n"
        "Behavioral NLP Framework — Cialdini Principle Analysis",
        pad=15
    )
    ax.set_xlabel("Cialdini Persuasion Principles", labelpad=10)
    ax.set_ylabel("Department", labelpad=10)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Heatmap saved → {save_path}")


# ------------------------------------------------------------------
# 2. Vulnerability Network Graph
# ------------------------------------------------------------------

def plot_network_graph(G: nx.DiGraph, save_path: str = GRAPH_OUTPUT_FILE):
    """
    Plots the NetworkX vulnerability graph.

    Node colors:
        - Red circles: high-risk employees
        - Orange circles: medium-risk employees
        - Green circles: low-risk employees
        - Blue squares: departments
        - Purple diamonds: Cialdini principles

    Edge colors:
        - Dark red: HIGH risk connection
        - Orange: MEDIUM risk connection
        - Gray: membership edge

    Args:
        G: The DiGraph from graph_builder.build_vulnerability_graph()
        save_path: File path to save the figure.
    """
    _ensure_output_dir()

    fig, ax = plt.subplots(figsize=(18, 12))

    # ---------------------------------------------------------------
    # Separate nodes by type for different rendering
    # ---------------------------------------------------------------
    employee_nodes  = [n for n, d in G.nodes(data=True) if d.get("node_type") == "employee"]
    dept_nodes      = [n for n, d in G.nodes(data=True) if d.get("node_type") == "department"]
    principle_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "principle"]

    # ---------------------------------------------------------------
    # Layout: spring layout with fixed seed for reproducibility
    # ---------------------------------------------------------------
    pos = nx.spring_layout(G, seed=42, k=2.5)

    # ---------------------------------------------------------------
    # Color employees by overall risk score
    # ---------------------------------------------------------------
    def _employee_color(node):
        risk = G.nodes[node].get("overall_risk", 0.0)
        if risk >= 0.65:
            return RISK_COLORS["HIGH"]
        elif risk >= 0.40:
            return RISK_COLORS["MEDIUM"]
        else:
            return RISK_COLORS["LOW"]

    emp_colors = [_employee_color(n) for n in employee_nodes]

    # ---------------------------------------------------------------
    # Draw nodes
    # ---------------------------------------------------------------
    nx.draw_networkx_nodes(
        G, pos, nodelist=employee_nodes,
        node_color=emp_colors, node_size=300,
        node_shape="o", alpha=0.85, ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=dept_nodes,
        node_color="#3498DB", node_size=700,
        node_shape="s", alpha=0.9, ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=principle_nodes,
        node_color="#9B59B6", node_size=600,
        node_shape="D", alpha=0.9, ax=ax
    )

    # ---------------------------------------------------------------
    # Draw edges
    # ---------------------------------------------------------------
    membership_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "member_of"]
    high_edges   = [(u, v) for u, v, d in G.edges(data=True) if d.get("risk_tier") == "HIGH"]
    medium_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("risk_tier") == "MEDIUM"]

    nx.draw_networkx_edges(G, pos, edgelist=membership_edges,
        edge_color="#BDC3C7", alpha=0.3, width=0.8, ax=ax, arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=high_edges,
        edge_color=RISK_COLORS["HIGH"], alpha=0.7, width=1.8,
        ax=ax, arrows=True, arrowsize=12)
    nx.draw_networkx_edges(G, pos, edgelist=medium_edges,
        edge_color=RISK_COLORS["MEDIUM"], alpha=0.5, width=1.2,
        ax=ax, arrows=True, arrowsize=10)

    # ---------------------------------------------------------------
    # Labels (only departments and principles are labeled by default
    # to avoid clutter; employees show IDs)
    # ---------------------------------------------------------------
    dept_labels      = {n: n for n in dept_nodes}
    principle_labels = {n: n for n in principle_nodes}
    emp_labels       = {n: G.nodes[n].get("name", n).split()[0] for n in employee_nodes}

    nx.draw_networkx_labels(G, pos, labels=dept_labels,
        font_size=9, font_weight="bold", font_color="#1A252F", ax=ax)
    nx.draw_networkx_labels(G, pos, labels=principle_labels,
        font_size=8, font_weight="bold", font_color="#4A235A", ax=ax)
    nx.draw_networkx_labels(G, pos, labels=emp_labels,
        font_size=6, font_color="#2C3E50", ax=ax)

    # ---------------------------------------------------------------
    # Legend
    # ---------------------------------------------------------------
    legend_elements = [
        mpatches.Patch(color=RISK_COLORS["HIGH"],   label="Employee — High Risk"),
        mpatches.Patch(color=RISK_COLORS["MEDIUM"], label="Employee — Medium Risk"),
        mpatches.Patch(color=RISK_COLORS["LOW"],    label="Employee — Low Risk"),
        mpatches.Patch(color="#3498DB",              label="Department"),
        mpatches.Patch(color="#9B59B6",              label="Cialdini Principle"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9, framealpha=0.9)

    ax.set_title(
        "Organizational Vulnerability Network\n"
        "Edges indicate HIGH / MEDIUM susceptibility to persuasion principle",
        fontsize=14, fontweight="bold", pad=20
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Network graph saved → {save_path}")


# ------------------------------------------------------------------
# 3. Top Individuals Risk Bar Chart
# ------------------------------------------------------------------

def plot_top_risk_individuals(profiles: list, top_n: int = 10,
                               save_path: str = None):
    """
    Horizontal bar chart of the top N highest overall-risk
    individuals, colored by risk tier.
    """
    _ensure_output_dir()
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "top_risk_individuals.png")

    sorted_profiles = sorted(
        profiles,
        key=lambda p: p.get("overall_risk_score", 0.0),
        reverse=True
    )[:top_n]

    names  = [f"{p['name']} ({p['department']})" for p in sorted_profiles]
    scores = [p.get("overall_risk_score", 0.0) for p in sorted_profiles]
    colors = [
        RISK_COLORS["HIGH"] if s >= 0.65 else
        RISK_COLORS["MEDIUM"] if s >= 0.40 else
        RISK_COLORS["LOW"]
        for s in scores
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names[::-1], scores[::-1], color=colors[::-1], edgecolor="white", height=0.6)

    for bar, score in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}", va="center", ha="left", fontsize=9)

    ax.set_xlim(0, 1.15)
    ax.axvline(0.65, color=RISK_COLORS["HIGH"],   linestyle="--", linewidth=1, alpha=0.6, label="High threshold")
    ax.axvline(0.40, color=RISK_COLORS["MEDIUM"], linestyle="--", linewidth=1, alpha=0.6, label="Medium threshold")

    ax.set_xlabel("Overall Susceptibility Score")
    ax.set_title(f"Top {top_n} Highest-Risk Individuals\nOverall Social Engineering Susceptibility Score")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Top risk chart saved → {save_path}")


# ------------------------------------------------------------------
# 4. CSV Risk Report
# ------------------------------------------------------------------

def export_csv_report(profiles: list, save_path: str = REPORT_OUTPUT_FILE):
    """
    Exports a structured CSV report with one row per employee,
    showing OCEAN scores, susceptibility scores, and risk tiers.
    """
    _ensure_output_dir()
    rows = []

    for p in profiles:
        row = {
            "ID":         p.get("id"),
            "Name":       p.get("name"),
            "Department": p.get("department"),
            "Title":      p.get("title"),
        }

        ocean = p.get("ocean_scores", {})
        for trait in OCEAN_TRAITS:
            row[f"OCEAN_{trait}"] = round(ocean.get(trait, 0.0), 4)

        sus = p.get("susceptibility_scores", {})
        tiers = p.get("risk_tiers", {})
        for principle in CIALDINI_PRINCIPLES:
            row[f"Sus_{principle}"]  = round(sus.get(principle, 0.0), 4)
            row[f"Risk_{principle}"] = tiers.get(principle, "LOW")

        row["Overall_Risk_Score"] = p.get("overall_risk_score", 0.0)
        row["Top_Vulnerability"]  = (
            p["top_vulnerabilities"][0][0]
            if p.get("top_vulnerabilities") else "N/A"
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"[Visualizer] CSV report saved → {save_path}")
    return df
