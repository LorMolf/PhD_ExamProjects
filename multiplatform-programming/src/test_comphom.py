from comphom_wrapper import py_process_file, py_find_generators, py_create_matrix
from tqdm import tqdm
import os

def compute_boundary(simplex):
    """
    Computes the boundary of a simplex.
    For each vertex, removes it to form a face simplex with a coefficient.
    Alternating signs: +1, -1, +1, ...
    
    :param simplex: List of vertex indices, e.g., [1, 2, 3]
    :return: List of [c, simplex] pairs, e.g., [[1, [2, 3]], [-1, [1, 3]], [1, [1, 2]]]
    """
    boundary = []
    sign = 1
    n = len(simplex)
    for i in range(n):
        face = simplex[:i] + simplex[i+1:]
        boundary.append([sign, face])
        sign *= -1
    return boundary

def process_triangulation(triangulation):
    """
    Processes a single triangulation to compute homology groups.
    """
    w_prev = 0
    z_cur = 0
    b_prev = []
    
    if not triangulation:
        return "(0)\n"
    
    max_dim = len(triangulation[0])
    boundaries = []
    for simplex in triangulation:
        boundary = compute_boundary(simplex)
        boundaries.extend(boundary)
    
    result = "("
    
    for dim in range(max_dim):
        try:
            generators = py_find_generators(boundaries)
            if not generators:
                result += "0"
                if dim < max_dim - 1:
                    result += ","
                continue
            
            # Create matrix with actual data - no need for empty initialization
            matrix_wrapper = py_create_matrix(generators, boundaries)
            if matrix_wrapper is None:
                result += "0"
                if dim < max_dim - 1:
                    result += ","
                continue
            
            try:
                matrix_snf = matrix_wrapper.nf_smith()
            except Exception as e:
                print(f"SNF computation error: {e}")
                matrix_snf = None
            
            if matrix_snf is None:
                result += "0"
                if dim < max_dim - 1:
                    result += ","
                continue
                
            # Get homology information
            z_cur = matrix_snf.get_num_zero_cols()
            homology_number = z_cur - w_prev
            torsion = matrix_snf.get_torsion()
            
            result += str(homology_number)
            if torsion:
                for coeff in torsion:
                    result += f"+Z_{coeff}"
            
            if dim < max_dim - 1:
                result += ","
            
            w_prev = matrix_snf.get_num_non_zero_rows()
            
            # Update boundaries for next iteration
            boundaries = []
            for generator in generators:
                boundary = compute_boundary(generator)
                boundaries.extend(boundary)
                
        except Exception as e:
            print(f"Error processing dimension {dim}: {e}")
            result += "0"
            if dim < max_dim - 1:
                result += ","
            continue
    
    result += ")\n"
    return result

def test():
    """
    Main test function to process a triangulation file and compute homology groups.
    """
    input_file = "/app/multiplatform-programming/test_files/3d_manifold.ct"  # Update as needed
    output_file = "/app/multiplatform-programming/test_files/test.out"  # Update as needed
    
    # Process the input file
    print(f"Processing input file: {input_file}")
    data = py_process_file(input_file)
    
    # Open output file
    try:
        with open(output_file, 'w') as out:
            print("\nComputing homology groups...")
            print(f"Writing results to {output_file}")
            print("The following computations might take a while -- especially if")
            print("large triangulations are present. Small dots will serve as a")
            print("progress indicator.\n")
            print("Computing...")
            
            dots = 0
            for triangulation in tqdm(data, desc="Computing homology groups"):
                if not triangulation:
                    print("Warning: Detected empty triangulation. Ignoring...\n")
                    continue
                homology_info = process_triangulation(triangulation)
                out.write(homology_info)
                # Print progress dot
                #print(".", end='', flush=True)
                #dots += 1
                #if dots >= 80:
                #    print()
                #    dots = 0
            print("\n...finished.")
    except Exception as e:
        print(f"Error writing to output file: {e}")

if __name__ == "__main__":
    test()
