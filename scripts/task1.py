import pandas as pd
import igraph as ig 
import seaborn as sns
import numpy as np
import lzma
import seaborn
import os
from Bio import Align

def read_data(file_path):
    with lzma.open(file_path, 'rt', encoding='utf-8') as f:
        sequences = f.readlines()
    return(sequences)


def create_BLOSUM_similairty(sequences):
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
    aligner.mode = "global"
    nsequence = len(sequences)
    similarity_matrix = np.zeros((nsequence, nsequence))

    max_scores = np.zeros(nsequence)
    for i in range(nsequence):
        seq = sequences[i].strip()
        # S_i,i is the maximum score for sequence i
        max_scores[i] = aligner.score(seq, seq)
    
    for i in range(nsequence): 
        seq_i_cleaned = str(sequences[i]).strip()
        S_max_i = max_scores[i]
        for j in range(nsequence): 
            seq_j_cleaned = str(sequences[j]).strip()

            score = aligner.score(seq_i_cleaned, seq_j_cleaned)
            similarity_matrix[i, j] = score

    return similarity_matrix

def create_normalized_matrix(similarity_matrix):
    diag = np.diag(similarity_matrix)
    denominator = np.sqrt(np.outer(diag, diag))
    normalized_matrix = similarity_matrix / denominator

    return(normalized_matrix)

def create_graph(normalized_matrix):
    num_sequences = normalized_matrix.shape[0]
    edges = [(i, j) for i in range(num_sequences) for j in range(i + 1, num_sequences) if
                normalized_matrix[i, j] > 0.4]
    graph = ig.Graph()
    graph.add_vertices(num_sequences)
    graph.add_edges(edges)


    return graph

def community_detection_plotting(graph):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "../results/network_clusters.png")

    communities = graph.community_infomap()
    rainbow = seaborn.color_palette("husl", len(communities))
    colours = [rainbow[communities.membership[v]] for v in graph.vs.indices]
    graph.vs["color"] = colours
    ig.plot(graph, bbox=(1280, 1280), layout="fr", target = output_file)


if __name__ == "__main__":
    sequences = read_data("../data/sequences.txt.xz")
    print(f"Loaded {len(sequences)} sequences.")

    blosum_matrix = create_BLOSUM_similairty(sequences=sequences)
    print("Creating BLOsUM Similarity Matrix")

    normalized_matrix = create_normalized_matrix(blosum_matrix)
    print("Creating Normalized Matrix")

    graph = create_graph(normalized_matrix=normalized_matrix)
    print("Creating graph")

    communities = community_detection_plotting(graph)
    print("Communities Detected and file Saved")


