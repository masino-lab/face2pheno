import json
import csv # <<< ADDED: Import the CSV library
from itertools import combinations

def load_adjacency_list(filepath):
    """Loads the adjacency list from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            return {int(k): v for k, v in data.items()}
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{filepath}' is not a valid JSON file.")
        return None

def find_triangles(adjacency_list):
    """Finds all unique, non-overlapping triangles in a graph."""
    triangles = set()
    nodes = sorted(adjacency_list.keys())

    for u in nodes:
        neighbors_of_u = adjacency_list.get(u, [])
        if len(neighbors_of_u) < 2:
            continue
            
        for v, w in combinations(neighbors_of_u, 2):
            if w in adjacency_list.get(v, []):
                triangle = tuple(sorted((u, v, w)))
                triangles.add(triangle)

    return sorted(list(triangles))

def find_adjacent_triangles(indexed_triangles):
    """Finds all pairs of triangles that share a common edge."""
    edge_to_triangle_map = {}
    
    for tri_index, nodes in indexed_triangles.items():
        edges = [
            tuple(sorted((nodes[0], nodes[1]))),
            tuple(sorted((nodes[1], nodes[2]))),
            tuple(sorted((nodes[2], nodes[0])))
        ]
        for edge in edges:
            if edge not in edge_to_triangle_map:
                edge_to_triangle_map[edge] = []
            edge_to_triangle_map[edge].append(tri_index)
            
    adjacent_pairs = []
    for edge, tri_indices in edge_to_triangle_map.items():
        if len(tri_indices) > 1:
            for pair in combinations(tri_indices, 2):
                adjacent_pairs.append(tuple(sorted(pair)))

    return sorted(list(set(adjacent_pairs)))

# --- NEW CSV SAVING FUNCTIONS ---

def save_triangles_to_csv(indexed_triangles, output_path):
    """Saves the indexed list of triangles to a CSV file.

    Args:
        indexed_triangles (dict): A map of triangle_index to its nodes.
        output_path (str): Path to the output CSV file.
    """
    header = ['triangle_index', 'node_1', 'node_2', 'node_3']
    
    try:
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for tri_index, nodes in indexed_triangles.items():
                writer.writerow([tri_index, nodes[0], nodes[1], nodes[2]])
        print(f"Successfully saved indexed triangles to {output_path}")
    except IOError as e:
        print(f"Error: Could not write to file {output_path}. Reason: {e}")

def save_adjacencies_to_csv(adjacent_pairs, output_path):
    """Saves the list of adjacent triangle pairs to a CSV file.

    Args:
        adjacent_pairs (list): A list of tuples, each with two triangle indices.
        output_path (str): Path to the output CSV file.
    """
    header = ['triangle_index_1', 'triangle_index_2']
    
    try:
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(adjacent_pairs)
        print(f"Successfully saved adjacent triangle pairs to {output_path}")
    except IOError as e:
        print(f"Error: Could not write to file {output_path}. Reason: {e}")


def main():
    """Main function to run the analysis and save the results."""
    filepath = 'face_mesh_adjacency_list.json'
    adjacency_list = load_adjacency_list(filepath)

    if not adjacency_list:
        return

    # 1. Find all triangles
    triangles = find_triangles(adjacency_list)

    # 2. Assign a unique index to each triangle
    indexed_triangles = {i: triangle for i, triangle in enumerate(triangles)}

    # 3. Find triangles that share an edge
    adjacent_triangle_pairs = find_adjacent_triangles(indexed_triangles)

    # --- Print summaries to console ---
    print(f"Found {len(indexed_triangles)} unique triangles.")
    print(f"Found {len(adjacent_triangle_pairs)} adjacent triangle pairs.\n")
    
    # --- Save the results to CSV files ---
    save_triangles_to_csv(indexed_triangles, 'triangles_list.csv')
    save_adjacencies_to_csv(adjacent_triangle_pairs, 'adjacent_triangles.csv')


if __name__ == "__main__":
    main()