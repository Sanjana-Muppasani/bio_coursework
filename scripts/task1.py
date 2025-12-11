import pandas as pd
import igraph as ig 
import seaborn as sns
import numpy as np
import lzma
import os
import random
from Bio import Align
from Bio.Align import substitution_matrices

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    ig.set_random_number_generator(random)

def read_data(file_path):
    with lzma.open(file_path, 'rt', encoding='utf-8') as f:
        sequences = [line.strip() for line in f.readlines() if line.strip()]
    return sequences

def compute_distance_matrix(sequences):
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    aligner.mode = 'global'
    
    num_sequences = len(sequences)
    dist_matrix = np.zeros((num_sequences, num_sequences))
    
    self_scores = [aligner.score(s, s) for s in sequences]

    for i in range(num_sequences):
        dist_matrix[i, i] = 0.0 
        
        for j in range(i + 1, num_sequences):
            raw_score = aligner.score(sequences[i], sequences[j])
            
            # Normalize
            denom = np.sqrt(self_scores[i] * self_scores[j])
            norm_sim = raw_score / denom if denom > 0 else 0
            distance = 1.0 - norm_sim
            distance = max(0.0, distance)
            
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance
            
    return dist_matrix

def create_graph(distance_matrix, threshold):
    num_sequences = distance_matrix.shape[0]
    edges = [(i, j) for i in range(num_sequences) for j in range(i + 1, num_sequences) if
             distance_matrix[i, j] < threshold]
             
    graph = ig.Graph()
    graph.add_vertices(num_sequences)
    graph.add_edges(edges)

    return graph

def get_global_metrics_table(graph):
    if graph.vcount() == 0:
        return pd.DataFrame()
        
    metrics = {
        "Node_Count": graph.vcount(),
        "Edge_Count": graph.ecount(),
        "Density": graph.density(), 
        "Transitivity_Global": graph.transitivity_undirected(),
        "Diameter": graph.diameter() if graph.ecount() > 0 else 0,
        "Average_Path_Length": graph.average_path_length() if graph.ecount() > 0 else 0,
        "Connected_Components": len(graph.connected_components())
    }
    
    df = pd.DataFrame([metrics])
    return df

def community_detection_plotting(graph, output_file):
    if graph.vcount() == 0:
        print("Empty graph, skipping plot.")
        return
        
    np.random.seed(42)  
    communities = graph.community_infomap() 
    
    rainbow = sns.color_palette("husl", len(communities))
    colours = [rainbow[communities.membership[v]] for v in graph.vs.indices]
    graph.vs["color"] = colours
    
    ig.plot(graph, bbox=(1280, 1280), layout="fr", target=output_file)
    return communities
def generate_threshold_table(distance_matrix, thresholds):
    results = []
    
    print(f"\n{'='*60}")
    print(f"Generating Network Topology Table (Distance Logic)")
    print(f"{'='*60}")

    for t in thresholds:
        num_nodes = distance_matrix.shape[0]
        edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes) 
                 if distance_matrix[i, j] < t]
        
        g = ig.Graph(n=num_nodes, edges=edges)
        comp = g.connected_components()
        
        if g.ecount() > 0:
            avg_path = g.average_path_length() 
        else:
            avg_path = 0.0

        metrics = {
            "Threshold": t,
            "Edge Count": g.ecount(),
            "Density": f"{g.density():.6f}", 
            "Connected Components": len(comp),
            "Avg. Path Length": f"{avg_path:.6f}"
        }
        results.append(metrics)
    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    set_global_seed(42)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    data_dir = os.path.join(script_dir, '..', 'data')
    results_dir = os.path.join(script_dir, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    input_file = os.path.join(data_dir, "sequences.txt.xz")
    output_image = os.path.join(results_dir, "network_clusters.png")

    print(f"Reading data from: {input_file}")
    sequences = read_data(input_file)
    print(f"Loaded {len(sequences)} sequences.")

    print("Computing Distance Matrix (BLOSUM62)...")
    distance_matrix = compute_distance_matrix(sequences=sequences)

    print(f"Minimum Distance: {np.min(distance_matrix)}")
    print(f"Maximum Distance: {np.max(distance_matrix)}")

    test_thresholds = [0.4, 0.6, 0.7, 0.76, 0.8, 0.9]
      
    topology_table = generate_threshold_table(distance_matrix, test_thresholds)
    print(topology_table.to_string(index=False))

    chosen_threshold = 0.76
    
    print(f"Creating graph with Distance Threshold < {chosen_threshold}...")
    graph_optimal = create_graph(distance_matrix=distance_matrix, threshold=chosen_threshold)
    
    print("Detecting communities...")
    communities = community_detection_plotting(graph_optimal, output_image)
    print(f"Saved plot to {output_image}")

    graph_table = get_global_metrics_table(graph_optimal)
    print("-----Graph Properties Table-----")
    print(graph_table)