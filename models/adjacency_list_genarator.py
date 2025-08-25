import json
from collections import defaultdict
import mediapipe as mp

def build_adjacency_list(connections):
    """Builds an adjacency list from a MediaPipe connection set.

    An adjacency list is a dictionary where each key is a landmark index and the
    value is a set of its neighboring indices.

    Args:
        connections: A set of frozenset pairs from MediaPipe (e.g.,
                     mp.solutions.face_mesh.FACEMESH_TESSELATION).

    Returns:
        A collections.defaultdict of sets representing the adjacency list.
    """
    adjacency_list = defaultdict(set)
    for connection in connections:
        p1, p2 = connection
        adjacency_list[p1].add(p2)
        adjacency_list[p2].add(p1)
    return adjacency_list


def save_adjacency_list_to_json(adjacency_list, output_path):
    """Saves the adjacency list dictionary to a formatted JSON file.

    Args:
        adjacency_list: The dictionary of neighbors, typically the output from
                        build_adjacency_list.
        output_path: The file path for the output JSON file.
    """

    json_compatible_list = {
        node: sorted(list(neighbors))
        for node, neighbors in adjacency_list.items()
    }

    try:
        with open(output_path, 'w') as f:
            json.dump(json_compatible_list, f, indent=4)
        print(f"Successfully saved adjacency list to {output_path}")
    except IOError as e:
        print(f"Error: Could not write to file {output_path}. Reason: {e}")


def main():
    """
    Generates and saves the adjacency list for the MediaPipe Face Mesh topology.
    """

    face_connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
    output_filename = 'face_mesh_adjacency_list.json'

    print("Building adjacency list from MediaPipe Face Mesh connections...")
    adjacency_list = build_adjacency_list(face_connections)

    save_adjacency_list_to_json(adjacency_list, output_filename)


if __name__ == "__main__":
    main()