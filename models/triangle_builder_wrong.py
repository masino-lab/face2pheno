import json
import csv
import random  
from itertools import combinations
from collections import Counter  

# --- UPDATED: Map of region names to numerical group IDs ---
# "Other" is a separate group (9). "Mixed" is now a randomly-handled condition.
REGION_TO_GROUP_ID = {
    "Iris": 0,
    "Left Eye": 1,
    "Right Eye": 2,
    "Left Eyebrow": 3,
    "Right Eyebrow": 4,
    "Nose": 5,
    "Lips (Inner)": 6,
    "Lips (Outer)": 7,
    "Face Contour": 8,
    "Other": 9,      # For landmarks not in any defined range
    "Unknown": 9     # Fallback, will be mapped to "Other"
}

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

# --- UPDATED: Supernode Grouping Functions ---

def get_landmark_region_id(landmark_index):
    """
    Assigns a landmark to a region ID based on the priority map.
    Ranges are prioritized (most specific/smallest first) to resolve overlaps.
    Returns an integer group ID.
    """
    
    # Priority 0: Iris
    if 468 <= landmark_index <= 477:
        return REGION_TO_GROUP_ID["Iris"]
        
    # Eyebrows
    if 46 <= landmark_index <= 65:
        return REGION_TO_GROUP_ID["Left Eyebrow"]
    if 276 <= landmark_index <= 295:
        return REGION_TO_GROUP_ID["Right Eyebrow"]
        
    # Eyes (must be AFTER Eyebrows)
    if 33 <= landmark_index <= 133:
        return REGION_TO_GROUP_ID["Left Eye"]
    if 263 <= landmark_index <= 362:
        return REGION_TO_GROUP_ID["Right Eye"]
    
    # Lips
    if 78 <= landmark_index <= 308:
        return REGION_TO_GROUP_ID["Lips (Inner)"]
    if 61 <= landmark_index <= 291:
        return REGION_TO_GROUP_ID["Lips (Outer)"]
    
    # Nose (must be AFTER eyes/lips)
    if 1 <= landmark_index <= 168:
        return REGION_TO_GROUP_ID["Nose"]
        
    # Face Contour (must be AFTER nose/eyes/lips)
    if (0 <= landmark_index <= 16) or (165 <= landmark_index <= 199):
        return REGION_TO_GROUP_ID["Face Contour"]
        
    # Default group if no range matches
    return REGION_TO_GROUP_ID["Other"]

def determine_triangle_group_id(triangle_nodes):
    """
    Determines the group ID of a triangle.
    - If a majority (2 or 3) nodes are in one group, it picks that group.
    - If all 3 nodes are in different groups ("Mixed"), it randomly
      chooses one of those three groups.
    """
    # Get the region ID for each of the 3 nodes
    region_ids = [get_landmark_region_id(node) for node in triangle_nodes]
    
    # Count the occurrences of each region ID
    region_counts = Counter(region_ids)
    
    # Find the most common region ID and its count
    most_common = region_counts.most_common(1)
    
    if not most_common:
        return REGION_TO_GROUP_ID["Unknown"] # Should not happen
    
    group_id, count = most_common[0]
    
    # If count is 2 or 3, a clear majority exists. Use that group.
    if count > 1:
        return group_id
    
    # If count is 1, all 3 nodes are different ("Mixed")
    # Randomly assign the triangle to one of the three node groups
    return random.choice(region_ids)

# --- CSV Saving Functions (Unchanged) ---

def save_triangles_to_csv(indexed_triangles, triangle_groups, output_path):
    """Saves the indexed list of triangles to a CSV file.

    Args:
        indexed_triangles (dict): A map of triangle_index to its nodes.
        triangle_groups (dict): A map of triangle_index to its supernode group ID.
        output_path (str): Path to the output CSV file.
    """
    header = ['triangle_index', 'node_1', 'node_2', 'node_3', 'supernode_group']
    
    try:
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for tri_index, nodes in indexed_triangles.items():
                # Get the pre-calculated group ID
                group = triangle_groups.get(tri_index, REGION_TO_GROUP_ID["Unknown"])
                writer.writerow([tri_index, nodes[0], nodes[1], nodes[2], group])
        print(f"Successfully saved indexed triangles to {output_path}")
    except IOError as e:
        print(f"Error: Could not write to file {output_path}. Reason: {e}")

def save_adjacencies_to_csv(adjacent_pairs, output_path):
    """Saves the list of adjacent triangle pairs to a CSV file."""
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
    filepath = '/home/abamini/face2pheno/data/processed/face_mesh_adjacency_list.json'
    adjacency_list = load_adjacency_list(filepath)

    if not adjacency_list:
        return

    # 1. Find all triangles
    triangles = find_triangles(adjacency_list)

    # 2. Assign a unique index to each triangle
    indexed_triangles = {i: triangle for i, triangle in enumerate(triangles)}

    # 3. Determine numerical group for each triangle
    print("Determining supernode group ID for each triangle...")
    triangle_groups = {}
    for tri_index, nodes in indexed_triangles.items():
        triangle_groups[tri_index] = determine_triangle_group_id(nodes)
    print("...Done determining groups.")

    # 4. Find triangles that share an edge
    adjacent_triangle_pairs = find_adjacent_triangles(indexed_triangles)

    # --- Print summaries to console ---
    print(f"Found {len(indexed_triangles)} unique triangles.")
    print(f"Found {len(adjacent_triangle_pairs)} adjacent triangle pairs.\n")
    
    # --- Save the results to CSV files ---
    save_triangles_to_csv(indexed_triangles, triangle_groups, 'triangles_list.csv')
    save_adjacencies_to_csv(adjacent_triangle_pairs, 'adjacent_triangles.csv')


if __name__ == "__main__":
    main()