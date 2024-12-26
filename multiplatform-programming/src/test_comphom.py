from comphom_wrapper import py_process_file, py_find_generators, py_create_matrix

# def test():
#     # Test `py_process_file`
#     data = py_process_file("/app/multiplatform-programming/test_files/test.ct")
#     print("Processed data:", data)

#     # Test `py_find_generators`
#     chains = [
#         [[1, [1, 2]], [-1, [2, 3]]],  # Example chains
#     ]
#     generators = py_find_generators(chains)
#     print("Generators:", generators)

#     # Test `py_create_matrix`
#     generators = [[1, 2], [2, 3]]  # Example simplices
#     boundaries = [
#         [[1, [1, 2]], [-1, [2, 3]]],  # Example chains
#     ]
#     matrix = py_create_matrix(generators, boundaries)
#     matrix.print()

# if __name__ == "__main__":
#     test()

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
    
    :param triangulation: List of simplices, each simplex is a list of vertex indices.
    :return: Homology group information as a string.
    """
    w_prev = 0
    z_cur = 0
    b_prev = []
    
    if not triangulation:
        return ""
    
    # Get max_dim from first simplex
    max_dim = len(triangulation[0])
    
    # Compute initial boundaries
    boundaries = []
    for simplex in triangulation:
        boundary = compute_boundary(simplex)
        boundaries.extend(boundary)
    
    result = "("
    
    for dim in range(max_dim):
        # Find generators
        generators = py_find_generators(boundaries)
        
        # Create boundary matrix
        matrix_wrapper = py_create_matrix(generators, boundaries)
        
        # Compute Smith Normal Form (SNF)
        matrix_snf = matrix_wrapper.nf_smith()
        
        if matrix_snf is None:
            # Handle case where SNF could not be computed
            homology_number = 0
            torsion = []
        else:
            # Get number of zero columns in SNF
            z_cur = matrix_snf.get_num_zero_cols()
            
            # Compute homology number: z_cur - w_prev
            homology_number = z_cur - w_prev
            
            # Get torsion coefficients
            torsion = matrix_snf.get_torsion()
        
        # Append homology number
        result += str(homology_number)
        
        # Append torsion coefficients, if any
        if torsion:
            for coeff in torsion:
                result += f"+Z_{coeff}"
        
        # Append comma if not the last dimension
        if dim < max_dim - 1:
            result += ","
        
        # Update w_prev and b_prev for the next dimension
        if matrix_snf is not None:
            w_prev = matrix_snf.get_num_non_zero_rows()
            b_prev = torsion
        else:
            w_prev = 0
            b_prev = []
        
        # Prepare boundaries for the next dimension by computing boundaries of generators
        boundaries = []
        for generator in generators:
            boundary = compute_boundary(generator)
            boundaries.extend(boundary)
    
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
    print("Processed data:", data)
    
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
            for triangulation in data:
                print("***")
                print("Triangulation:")
                print(triangulation)
                print("**")
                if not triangulation:
                    print("Warning: Detected empty triangulation. Ignoring...\n")
                    continue
                homology_info = process_triangulation(triangulation)
                out.write(homology_info)
                # Print progress dot
                print(".", end='', flush=True)
                dots += 1
                if dots >= 80:
                    print()
                    dots = 0
            print("\n...finished.")
    except Exception as e:
        print(f"Error writing to output file: {e}")

if __name__ == "__main__":
    test()
