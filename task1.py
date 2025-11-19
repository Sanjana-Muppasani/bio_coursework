import lzma
import os

import numpy as np
from Bio.Align import PairwiseAligner
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import igraph as ig
import matplotlib.pyplot as plt


def load_sequences(file_path):
    with lzma.open(file_path, "rt") as f:
        return [SeqRecord(Seq(line.strip()), id=f"seq{i + 1}") for i, line in enumerate(f) if line.strip()]


def compute_distance_matrix(sequences):
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    num_sequences = len(sequences)
    distance_matrix = np.zeros((num_sequences, num_sequences))
    for i in range(num_sequences):
        for j in range(i + 1, num_sequences):
            score = aligner.score(sequences[i].seq, sequences[j].seq)
            distance_matrix[i, j] = score
            distance_matrix[j, i] = score
    return distance_matrix


def construct_network(distance_matrix, threshold):
    num_sequences = distance_matrix.shape[0]
    edges = [(i, j) for i in range(num_sequences) for j in range(i + 1, num_sequences) if
             distance_matrix[i, j] > threshold]
    graph = ig.Graph()
    graph.add_vertices(num_sequences)
    graph.add_edges(edges)
    return graph


def visualize_plain_network(graph, layout, output_file="network_clusters.png"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "../results/network_clusters.png")

    visual_style = {
        "vertex_size": 10,
        "vertex_color": "blue",
        "vertex_label": None,
        "edge_width": 1,
        "layout": layout,
        "bbox": (800, 800),
        "margin": 20
    }
    ig.plot(graph, target=output_file, **visual_style)
    print(f"Plain network visualization saved as {output_file}")


def cluster_and_visualize_with_colors(graph, layout, output_file="network_clusters_coloured.png"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "../results/network_clusters_coloured.png")

    clusters = graph.community_multilevel()
    membership = clusters.membership
    num_clusters = len(set(membership))
    colormap = plt.colormaps["tab20"].resampled(num_clusters)
    colors = [colormap(i / num_clusters) for i in membership]
    visual_style = {
        "vertex_size": 10,
        "vertex_color": [f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}" for r, g, b, _ in colors],
        "vertex_label": None,
        "edge_width": 2,
        "layout": layout,
        "bbox": (800, 800),
        "margin": 20
    }
    ig.plot(graph, target=output_file, **visual_style)
    print(f"Clustered network visualization with colors saved as {output_file}")
    return clusters


def analyze_network(graph):
    return {
        "Number of Nodes": graph.vcount(),
        "Number of Edges": graph.ecount(),
        "Average Degree": sum(graph.degree()) / graph.vcount(),
        "Clustering Coefficient": graph.transitivity_undirected(),
        "Number of Connected Components": len(graph.components())
    }


def evaluate_thresholds(distance_matrix, thresholds):
    results = []
    for threshold in thresholds:
        graph = construct_network(distance_matrix, threshold)
        num_edges = graph.ecount()
        avg_degree = sum(graph.degree()) / graph.vcount() if graph.vcount() > 0 else 0
        num_components = len(graph.components())
        largest_component_size = max(len(c) for c in graph.components()) if num_components > 0 else 0
        clusters = graph.community_multilevel() if num_edges > 0 else None
        modularity = graph.modularity(clusters) if clusters else 0
        edge_density = graph.ecount() / (graph.vcount() * (graph.vcount() - 1) / 2) if graph.vcount() > 1 else 0
        results.append({
            "Threshold": threshold,
            "Edges": num_edges,
            "Avg Degree": avg_degree,
            "Components": num_components,
            "Largest Component": largest_component_size,
            "Modularity": modularity,
            "Edge Density": edge_density
        })
    return results


def plot_threshold_impact(results):
    thresholds = [res["Threshold"] for res in results]
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 3, 1)
    plt.plot(thresholds, [res["Edges"] for res in results], marker="o", label="Edges")
    plt.xlabel("Threshold")
    plt.ylabel("Edges")
    plt.title("Edges vs Threshold")
    plt.grid()

    plt.subplot(2, 3, 2)
    plt.plot(thresholds, [res["Components"] for res in results], marker="o", label="Components", color="red")
    plt.xlabel("Threshold")
    plt.ylabel("Components")
    plt.title("Components vs Threshold")
    plt.grid()

    plt.subplot(2, 3, 3)
    plt.plot(thresholds, [res["Largest Component"] for res in results], marker="o", label="Largest Component",
             color="green")
    plt.xlabel("Threshold")
    plt.ylabel("Largest Component Size")
    plt.title("Largest Component vs Threshold")
    plt.grid()

    plt.subplot(2, 3, 4)
    plt.plot(thresholds, [res["Modularity"] for res in results], marker="o", label="Modularity", color="orange")
    plt.xlabel("Threshold")
    plt.ylabel("Modularity")
    plt.title("Modularity vs Threshold")
    plt.grid()

    plt.subplot(2, 3, 5)
    plt.plot(thresholds, [res["Edge Density"] for res in results], marker="o", label="Edge Density", color="purple")
    plt.xlabel("Threshold")
    plt.ylabel("Edge Density")
    plt.title("Edge Density vs Threshold")
    plt.grid()

    plt.tight_layout()

    output_file = "../results/threshold_impact.png"
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Plot saved as {output_file}")


if __name__ == "__main__":

    sequences = load_sequences("../data/sequences.txt.xz")
    print(f"Loaded {len(sequences)} sequences.")

    print("Computing pairwise distance matrix...")
    distance_matrix = compute_distance_matrix(sequences)
    print("Pairwise distance matrix computed.")

    thresholds = np.linspace(10, np.max(distance_matrix), num=20)
    print("Evaluating thresholds...")
    results = evaluate_thresholds(distance_matrix, thresholds)

    for result in results:
        print(
            f"Threshold: {result['Threshold']:.2f}, "
            f"Edges: {result['Edges']}, "
            f"Avg Degree: {result['Avg Degree']:.2f}, "
            f"Components: {result['Components']}, "
            f"Largest Component: {result['Largest Component']}, "
            f"Modularity: {result['Modularity']:.2f}, "
            f"Edge Density: {result['Edge Density']:.4f}"
        )

    plot_threshold_impact(results)

    chosen_threshold = 120.53
    print(f"\nUsing chosen threshold: {chosen_threshold}")
    network = construct_network(distance_matrix, chosen_threshold)

    network_properties = analyze_network(network)
    for prop, value in network_properties.items():
        print(f"{prop}: {value}")

    layout = network.layout("fr")  # Generate layout once
    visualize_plain_network(network, layout)  # Reuse the layout
    clusters = cluster_and_visualize_with_colors(network, layout)

    print(f"Detected {len(clusters)} clusters.")
