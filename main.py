# main.py
#
# Entry point for the Behavioral NLP Framework.
# Run this file to execute the full pipeline end to end.
#
# What happens when you run this:
#   1. Synthetic organization profiles are generated
#   2. Personality (OCEAN) is inferred per profile
#   3. Susceptibility scores are computed for each Cialdini principle
#   4. A vulnerability graph is built using NetworkX
#   5. Visualizations are saved to the output/ directory
#   6. A summary report is printed and saved as CSV

import os
import sys
import json
import argparse
import time

# ------------------------------------------------------------------
# Make sure the project root is on the Python path
# (allows running from any directory)
# ------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_DIR
from data.synthetic_generator import generate_organization
from pipeline.personality_inference import _load_model, infer_batch
from pipeline.susceptibility_scorer import score_organization, aggregate_department_risk
from pipeline.graph_builder import build_vulnerability_graph, get_graph_summary, export_graph_json
from visualization.visualizer import (
    plot_heatmap,
    plot_network_graph,
    plot_top_risk_individuals,
    export_csv_report,
)


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║   Behavioral NLP Framework for Social Engineering Susceptibility  ║
║   Psycholinguistic Cognitive Vulnerability Modeling               ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_summary(profiles: list, graph_summary: dict):
    """Prints a readable summary of results to the terminal."""
    total = len(profiles)
    high_risk   = sum(1 for p in profiles if p.get("overall_risk_score", 0) >= 0.65)
    medium_risk = sum(1 for p in profiles if 0.40 <= p.get("overall_risk_score", 0) < 0.65)
    low_risk    = total - high_risk - medium_risk

    print("\n" + "="*65)
    print("  ORGANIZATIONAL RISK SUMMARY")
    print("="*65)
    print(f"  Total Employees Analyzed : {total}")
    print(f"  High Risk Individuals    : {high_risk}  ({round(high_risk/total*100)}%)")
    print(f"  Medium Risk Individuals  : {medium_risk}  ({round(medium_risk/total*100)}%)")
    print(f"  Low Risk Individuals     : {low_risk}  ({round(low_risk/total*100)}%)")
    print(f"\n  Most Targeted Principle  : {graph_summary['most_targeted_principle']}")
    print(f"  (by {graph_summary['most_targeted_count']} employees)")
    print(f"\n  Highest Risk Individual  : {graph_summary['highest_risk_employee']}")
    print(f"  Risk Score               : {graph_summary['highest_risk_score']}")
    print("="*65)

    print("\n  TOP 5 HIGHEST RISK EMPLOYEES:")
    print("  " + "-"*55)
    sorted_profiles = sorted(profiles, key=lambda p: p.get("overall_risk_score", 0), reverse=True)[:5]
    for i, p in enumerate(sorted_profiles, 1):
        top_vuln = p["top_vulnerabilities"][0][0] if p.get("top_vulnerabilities") else "N/A"
        tier     = p["top_vulnerabilities"][0][2] if p.get("top_vulnerabilities") else "N/A"
        print(
            f"  {i}. {p['name']:<22} | {p['department']:<12} | "
            f"Score: {p['overall_risk_score']:.2f} | "
            f"Top Lure: {top_vuln} [{tier}]"
        )

    print("\n  DEPARTMENT RISK OVERVIEW:")
    print("  " + "-"*55)
    dept_risks = {}
    for p in profiles:
        dept = p.get("department", "Unknown")
        dept_risks.setdefault(dept, []).append(p.get("overall_risk_score", 0))

    for dept, scores in sorted(dept_risks.items(), key=lambda x: -sum(x[1])/len(x[1])):
        avg = sum(scores) / len(scores)
        bar = "█" * int(avg * 20)
        print(f"  {dept:<14} | {bar:<20} | {avg:.2f}")

    print("\n  OUTPUT FILES:")
    print(f"  → output/susceptibility_heatmap.png")
    print(f"  → output/vulnerability_graph.png")
    print(f"  → output/top_risk_individuals.png")
    print(f"  → output/risk_report.csv")
    print(f"  → output/graph.json")
    print("="*65 + "\n")


def run_pipeline(num_employees: int = 30, use_transformer: bool = True):
    """
    Runs the full analysis pipeline.

    Args:
        num_employees: Number of synthetic profiles to generate.
        use_transformer: If True, attempts to load the HuggingFace
                         model. Set to False to skip to heuristic
                         mode (faster, no internet required).
    """
    start_time = time.time()
    print_banner()

    # ------------------------------------------------------------------
    # Step 1: Load Transformer Model (if requested)
    # ------------------------------------------------------------------
    if use_transformer:
        print("[Step 1/5] Loading personality inference model...")
        _load_model()
    else:
        print("[Step 1/5] Skipping transformer model (heuristic mode).")

    # ------------------------------------------------------------------
    # Step 2: Generate Synthetic Organization
    # ------------------------------------------------------------------
    print(f"\n[Step 2/5] Generating {num_employees} synthetic employee profiles...")
    profiles = generate_organization(
        num_employees=num_employees,
        save_path=os.path.join(OUTPUT_DIR, "synthetic_profiles.json")
    )
    print(f"  Generated {len(profiles)} profiles across {len(set(p['department'] for p in profiles))} departments.")

    # ------------------------------------------------------------------
    # Step 3: Personality Inference
    # ------------------------------------------------------------------
    print("\n[Step 3/5] Running personality inference (OCEAN scoring)...")
    profiles = infer_batch(profiles)

    method_used = profiles[0].get("inference_method", "unknown") if profiles else "unknown"
    print(f"  Inference complete. Method used: [{method_used}]")

    # ------------------------------------------------------------------
    # Step 4: Susceptibility Scoring
    # ------------------------------------------------------------------
    print("\n[Step 4/5] Computing social engineering susceptibility scores...")
    profiles = score_organization(profiles)

    dept_averages = aggregate_department_risk(profiles)
    print(f"  Susceptibility scoring complete for {len(profiles)} employees.")

    # ------------------------------------------------------------------
    # Step 5: Graph Building and Visualization
    # ------------------------------------------------------------------
    print("\n[Step 5/5] Building vulnerability graph and generating visualizations...")

    G = build_vulnerability_graph(profiles)
    graph_summary = get_graph_summary(G)

    export_graph_json(G, os.path.join(OUTPUT_DIR, "graph.json"))
    plot_heatmap(dept_averages)
    plot_network_graph(G)
    plot_top_risk_individuals(profiles)
    export_csv_report(profiles)

    # ------------------------------------------------------------------
    # Final Summary
    # ------------------------------------------------------------------
    elapsed = round(time.time() - start_time, 1)
    print(f"\n  Pipeline completed in {elapsed}s.")
    print_summary(profiles, graph_summary)

    return profiles, G


# ------------------------------------------------------------------
# CLI Entry Point
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Behavioral NLP Framework — Social Engineering Susceptibility Analyzer"
    )
    parser.add_argument(
        "--employees",
        type=int,
        default=30,
        help="Number of synthetic employee profiles to generate (default: 30)"
    )
    parser.add_argument(
        "--no-transformer",
        action="store_true",
        help="Skip HuggingFace model and use heuristic mode (faster, no download needed)"
    )
    args = parser.parse_args()

    run_pipeline(
        num_employees=args.employees,
        use_transformer=not args.no_transformer
    )
