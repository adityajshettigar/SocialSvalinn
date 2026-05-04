# pipeline/graph_builder.py
#
# Builds a NetworkX vulnerability graph from scored profiles.
#
# Graph structure:
#   - Nodes: individual employees + department nodes
#   - Edges: employee → department (membership)
#             employee → principle (if HIGH risk only)
#   - Node attributes: risk scores, OCEAN values, department
#   - Edge attributes: susceptibility score, risk tier
#
# This graph can later be analyzed (e.g., finding the highest-risk
# path into the organization) or exported for visualization.

import networkx as nx
import json
from config import CIALDINI_PRINCIPLES, DEPARTMENTS, RISK_THRESHOLDS


def build_vulnerability_graph(profiles: list) -> nx.DiGraph:
    """
    Constructs the organizational vulnerability graph.

    Node types:
        - "employee": individual profile node
        - "department": department aggregate node
        - "principle": Cialdini persuasion principle node

    Edge types:
        - employee → department: "member_of"
        - employee → principle: "susceptible_to" (only for HIGH risk)

    Args:
        profiles: Fully scored list of profile dicts.

    Returns:
        A directed NetworkX graph (DiGraph).
    """
    G = nx.DiGraph()

    # ---------------------------------------------------------------
    # Add Cialdini principle nodes
    # ---------------------------------------------------------------
    for principle in CIALDINI_PRINCIPLES:
        G.add_node(
            principle,
            node_type="principle",
            label=principle,
        )

    # ---------------------------------------------------------------
    # Add department nodes
    # ---------------------------------------------------------------
    for dept in DEPARTMENTS:
        G.add_node(
            dept,
            node_type="department",
            label=dept,
        )

    # ---------------------------------------------------------------
    # Add employee nodes and their edges
    # ---------------------------------------------------------------
    for profile in profiles:
        emp_id = profile["id"]
        dept = profile.get("department", "Unknown")

        G.add_node(
            emp_id,
            node_type="employee",
            name=profile.get("name", ""),
            department=dept,
            title=profile.get("title", ""),
            overall_risk=profile.get("overall_risk_score", 0.0),
            ocean=json.dumps(profile.get("ocean_scores", {})),
            top_vulnerability=(
                profile["top_vulnerabilities"][0][0]
                if profile.get("top_vulnerabilities") else "None"
            ),
        )

        # Employee → Department edge
        G.add_edge(emp_id, dept, edge_type="member_of")

        # Employee → Principle edges (Filtered for Extremes - HIGH risk only)
        sus_scores = profile.get("susceptibility_scores", {})
        risk_tiers = profile.get("risk_tiers", {})

        for principle, score in sus_scores.items():
            tier = risk_tiers.get(principle, "LOW")
            # Only draw edges for HIGH risk to eliminate the visual hairball
            if tier == "HIGH":
                G.add_edge(
                    emp_id,
                    principle,
                    edge_type="susceptible_to",
                    score=score,
                    risk_tier=tier,
                )

    return G


def get_graph_summary(G: nx.DiGraph) -> dict:
    """
    Computes basic graph-level statistics for reporting.

    Returns:
        Dict with node counts, edge counts, and high-risk stats.
    """
    employee_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "employee"]
    principle_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "principle"]

    high_risk_edges = [
        (u, v, d) for u, v, d in G.edges(data=True)
        if d.get("risk_tier") == "HIGH"
    ]

    # Most targeted principle (highest in-degree from employees)
    principle_in_degrees = {
        p: sum(1 for u, v in G.in_edges(p) if G.nodes[u].get("node_type") == "employee")
        for p in principle_nodes
    }
    most_targeted = max(principle_in_degrees, key=principle_in_degrees.get, default="None")

    # Highest risk employee
    emp_risks = {
        n: G.nodes[n].get("overall_risk", 0.0)
        for n in employee_nodes
    }
    highest_risk_emp = max(emp_risks, key=emp_risks.get, default="None")

    return {
        "total_employees": len(employee_nodes),
        "total_edges": G.number_of_edges(),
        "high_risk_edges": len(high_risk_edges),
        "most_targeted_principle": most_targeted,
        "most_targeted_count": principle_in_degrees.get(most_targeted, 0),
        "highest_risk_employee": G.nodes[highest_risk_emp].get("name", highest_risk_emp),
        "highest_risk_score": round(emp_risks.get(highest_risk_emp, 0.0), 4),
    }


def export_graph_json(G: nx.DiGraph, path: str):
    """
    Exports the graph as a JSON file (node-link format),
    which can be loaded into D3.js or Gephi for further analysis.
    """
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = nx.node_link_data(G)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[GraphBuilder] Graph exported to {path}")