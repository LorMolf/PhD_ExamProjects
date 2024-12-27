import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from comphom_wrapper import py_process_file, py_find_generators, py_create_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

def visualize_triangulation(triangulation, output_path):
    """
    Visualizes the simplicial complex represented by the triangulation.

    Args:
        triangulation (list): List of simplices, each represented as a list of vertex indices.
        output_path (str): Path to save the visualization.

    Visualization:
        - Points: Vertices of the simplices.
        - Lines: 1-dimensional edges of the simplices.
        - Polygons: 2-dimensional faces of the simplices (if available).
    """
    # Extract vertices
    vertices = {v for simplex in triangulation for v in simplex}
    edges = {tuple(sorted([simplex[i], simplex[j]])) 
             for simplex in triangulation for i in range(len(simplex)) for j in range(i + 1, len(simplex))}

    vertex_coords = {v: (v % 3, v // 3, (v // 3) % 2) for v in vertices}  # Simple pseudo-coordinates for visualization

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot vertices
    for v, coord in vertex_coords.items():
        ax.scatter(*coord, color='blue', s=50)
        ax.text(*coord, f"{v}", fontsize=8, color='black')

    # Plot edges
    for edge in edges:
        p1, p2 = edge
        coord1, coord2 = vertex_coords[p1], vertex_coords[p2]
        ax.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], [coord1[2], coord2[2]], color='gray', lw=1)

    # Plot faces
    for simplex in triangulation:
        if len(simplex) == 3:
            vertices_coords = [vertex_coords[v] for v in simplex]
            poly = Poly3DCollection([vertices_coords], alpha=0.5, edgecolor='r', linewidths=0.7)
            ax.add_collection3d(poly)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Simplicial Complex Visualization")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def visualize_homology_info(homology_info, output_path):
    """
    Visualizes the computed homology groups as a bar chart.

    Args:
        homology_info (list): List of homology group information for each dimension.
        output_path (str): Path to save the visualization.

    Visualization:
        - Dimensions are represented on the x-axis.
        - The rank of free parts is displayed as the height of the bar.
        - Torsion groups are annotated above each bar if present.
    """
    dimensions = list(range(len(homology_info)))
    ranks = []
    torsions = []

    for info in homology_info:
        if "+" in info:
            rank, torsion = info.split("+", 1)
            ranks.append(int(rank))
            torsions.append(torsion)
        else:
            ranks.append(int(info))
            torsions.append("")

    plt.figure(figsize=(10, 6))
    plt.bar(dimensions, ranks, color='skyblue', edgecolor='black')
    plt.xlabel("Dimension")
    plt.ylabel("Rank of Free Part")
    plt.title("Homology Groups Visualization")

    # Annotate torsion information above bars
    for i, torsion in enumerate(torsions):
        if torsion:
            plt.text(i, ranks[i] + 0.1, torsion, ha='center', va='bottom', fontsize=10, color='red')

    plt.xticks(dimensions)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def visualize_boundary_matrix(boundary_pairs, output_path, dimension, triangulation_index):
    """
    Visualizes the boundary matrix as a heatmap and saves the plot, only if valid boundary pairs exist.

    Args:
        boundary_pairs (list): List of boundary pairs for the simplicial complex.
        output_path (str): Path to save the visualization.
        dimension (int): Dimension of the simplices being visualized.
        triangulation_index (int): Index of the triangulation being processed.

    Returns:
        bool: True if the visualization was created, False otherwise.
    """
    if not boundary_pairs or all(not simplex for _, simplex in boundary_pairs):
        return False  # No valid simplices for this dimension, skip silently.

    all_indices = [idx for _, simplex in boundary_pairs for idx in simplex]
    if not all_indices:
        return False  # No valid indices, skip silently.

    matrix_size = max(all_indices) + 1
    matrix = np.zeros((matrix_size, matrix_size))

    for sign, simplex in boundary_pairs:
        for idx in simplex:
            matrix[idx, idx] = sign

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, cmap="coolwarm", cbar=False, xticklabels=False, yticklabels=False)
    plt.title(f"Boundary Matrix - Dimension {dimension}, Triangulation {triangulation_index}")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.savefig(output_path)
    plt.close()

    return True  # Visualization successfully created.



def save_homology_summary(homology_data, output_dir):
    """
    Saves the homology group summary for all triangulations to a CSV file.
    """
    summary_file = os.path.join(output_dir, "homology_summary.csv")
    with open(summary_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Triangulation Index", "Dimension", "Rank", "Torsion"])
        writer.writerows(homology_data)
    print(f"Homology summary saved to {summary_file}.")

def process_triangulation(triangulation, triangulation_index, output_dir):
    """
    Processes a single triangulation to compute homology groups and generate visualizations.

    Args:
        triangulation (list): List of simplices in the triangulation.
        triangulation_index (int): Index of the triangulation being processed.
        output_dir (str): Base directory for saving visualizations.

    Returns:
        str: Homology group information as a string.
    """
    if not triangulation:
        return "(0)\n"

    max_dim = len(triangulation[0])
    boundaries = []
    homology_info = []

    # Compute boundaries
    for simplex in triangulation:
        boundaries.extend(compute_boundary(simplex))

    # Save simplicial complex visualization
    visualize_triangulation(
        triangulation,
        os.path.join(output_dir, "simplicial_complex", f"complex_{triangulation_index}.png")
    )

    # Process each dimension
    for dim in range(max_dim):
        visualization_created = visualize_boundary_matrix(
            boundaries,
            os.path.join(output_dir, "boundary_matrices", f"boundary_matrix_dim_{dim}_{triangulation_index}.png"),
            dim,
            triangulation_index
        )

        if not visualization_created:
            continue  # Skip processing for this dimension if no visualization was created.

        generators = py_find_generators(boundaries)
        if not generators:
            homology_info.append("0")
            continue

        matrix_wrapper = py_create_matrix(generators, boundaries)
        if not matrix_wrapper:
            homology_info.append("0")
            continue

        try:
            matrix_snf = matrix_wrapper.nf_smith()
            z_cur = matrix_snf.get_num_zero_cols()
            rank = z_cur - len(homology_info)
            torsion = "+".join(f"Z_{t}" for t in matrix_snf.get_torsion())

            group_info = f"{rank}{'+' + torsion if torsion else ''}"
            homology_info.append(group_info)

            # Update boundaries for next dimension
            boundaries = [
                boundary
                for generator in generators
                for boundary in compute_boundary(generator)
            ]
        except:
            homology_info.append("0")
            continue

    # Save homology group visualization
    visualize_homology_info(
        homology_info,
        os.path.join(output_dir, "homology_groups", f"groups_{triangulation_index}.png")
    )

    return "(" + ",".join(homology_info) + ")\n"



def compute_boundary(simplex):
    """
    Computes the boundary of a simplex.
    
    Args:
        simplex (list): List of vertex indices representing the simplex.

    Returns:
        list: List of boundaries with alternating signs.
    """
    boundary = []
    sign = 1
    for i in range(len(simplex)):
        face = simplex[:i] + simplex[i + 1:]
        boundary.append([sign, face])
        sign *= -1
    return boundary

def test():
    """
    Main test function to process triangulation file, compute homology groups, and generate visualizations.

    The function:
    - Reads triangulations from the input file.
    - Processes each triangulation to compute homology groups.
    - Saves results and visualizations into structured subdirectories.
    """
    input_file = "/app/multiplatform-programming/test_files/2d_manifold.ct"
    output_file = "/app/multiplatform-programming/test_files/2d_manifold_test.out"
    output_dir = "/app/multiplatform-programming/test_files/plots/visualizations_2d_manifold"

    # Create structured output directories
    os.makedirs(os.path.join(output_dir, "simplicial_complex"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "boundary_matrices"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "homology_groups"), exist_ok=True)

    # Read and process triangulations
    print(f"Processing input file: {input_file}")
    data = py_process_file(input_file)[:10]
    print("Processing the first 10 triangulations.")

    try:
        with open(output_file, 'w') as out:
            print("\nComputing homology groups...")
            for i, triangulation in enumerate(tqdm(data, desc="Processing Triangulations")):
                if not triangulation:
                    continue
                homology_info = process_triangulation(triangulation, i, output_dir)
                out.write(homology_info)
            print("\nFinished processing.")
    except Exception as e:
        print(f"Error writing to output file: {e}")


if __name__ == "__main__":
    test()

