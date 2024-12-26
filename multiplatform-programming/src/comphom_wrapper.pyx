# cython: language_level=3

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
ctypedef unsigned long ulong

# Update simplex class and constructor
cdef extern from "simplex.h":
    cdef cppclass simplex:
        simplex()  # Add default constructor
        vector[ulong] vertices
        chain boundary()
        void print()
        bint operator==(simplex other)

cdef extern from "simplex.h":
    cdef cppclass chain_element:
        chain_element()  # Add default constructor
        long c
        simplex s

cdef extern from "simplex.h":
    cdef cppclass chain:
        chain()  # Add default constructor
        vector[chain_element] elements
        void print()

cdef extern from "matrix.h":
    cdef cppclass matrix:
        matrix()  # Default constructor
        matrix(const matrix& other)  # Copy constructor
        matrix assign(vector[long]&, ulong, ulong)
        matrix nf_smith()  # Add this line
        void transpose()
        void print()
        size_t get_num_rows()
        size_t get_num_cols()
        size_t get_num_zero_cols()
        size_t get_num_non_zero_rows()
        vector[ulong] get_torsion()

cdef extern from "comphom.h":
    vector[vector[simplex]] process_file(const char* filename)
    vector[simplex] find_generators(vector[chain] chains)
    size_t generator_position(vector[simplex] generators, simplex generator)
    matrix create_matrix(vector[simplex] generators, vector[chain] boundaries)

# Helper functions
cdef vector[ulong] pylist_to_vector_ulong(list py_list):
    cdef vector[ulong] c_vector
    for value in py_list:
        c_vector.push_back(value)
    return c_vector

cdef vector[simplex] pylist_to_vector_simplex(list py_list):
    cdef vector[simplex] c_vector
    cdef simplex s
    for item in py_list:
        s = simplex()  # Use default constructor
        s.vertices = pylist_to_vector_ulong(item)
        c_vector.push_back(s)
    return c_vector

cdef vector[chain] pylist_to_vector_chain(list py_list):
    cdef vector[chain] c_vector
    cdef chain c
    cdef chain_element ce
    #print(f"py_list: {py_list}")
    for item in py_list:
        c = chain()  # Use default constructor
        #for elem in item:
        elem = item
        ce = chain_element()  # Use default constructor
        ce.c = elem[0]
        ce.s.vertices = pylist_to_vector_ulong(elem[1])
        c.elements.push_back(ce)

        c_vector.push_back(c)
    return c_vector

# MatrixWrapper class
cdef class MatrixWrapper:
    cdef matrix* c_matrix

    def __cinit__(self, list generators, list boundaries):
        cdef vector[simplex] c_generators = pylist_to_vector_simplex(generators)
        cdef vector[chain] c_boundaries = pylist_to_vector_chain(boundaries)
        self.c_matrix = new matrix(create_matrix(c_generators, c_boundaries))

    def __dealloc__(self):
        if self.c_matrix != NULL:
            del self.c_matrix

    def print(self):
        if self.c_matrix != NULL:
            self.c_matrix.print()

    def get_num_rows(self):
        if self.c_matrix != NULL:
            return self.c_matrix.get_num_rows()
        return 0

    def get_num_cols(self):
        if self.c_matrix != NULL:
            return self.c_matrix.get_num_cols()
        return 0

    def nf_smith(self):
        """Compute Smith Normal Form of the matrix"""
        cdef MatrixWrapper wrapper
        cdef size_t rows, cols
        
        if self.c_matrix == NULL:
            print("Null matrix pointer")
            return None
            
        try:
            rows = self.c_matrix.get_num_rows()
            cols = self.c_matrix.get_num_cols()
            if rows == 0 or cols == 0:
                print(f"Invalid matrix dimensions: {rows}x{cols}")
                return None
                
            print(f"Computing SNF for {rows}x{cols} matrix...")
            
            # Create new wrapper and assign SNF result
            wrapper = MatrixWrapper([], [])
            wrapper.c_matrix = new matrix(self.c_matrix.nf_smith())
            
            if wrapper.c_matrix == NULL:
                print("Failed to create result matrix")
                return None
                
            return wrapper
            
        except Exception as e:
            print(f"SNF computation failed: {str(e)}")
            return None

    def get_num_zero_cols(self):
        """Get number of zero columns in the matrix"""
        if self.c_matrix != NULL:
            return self.c_matrix.get_num_zero_cols()
        return 0

    def get_num_non_zero_rows(self):
        """Get number of non-zero rows in the matrix"""
        if self.c_matrix != NULL:
            return self.c_matrix.get_num_non_zero_rows()
        return 0

    def get_torsion(self):
        """Get torsion coefficients from the matrix"""
        if self.c_matrix != NULL:
            return list(self.c_matrix.get_torsion())
        return []

# Python-accessible functions
def py_process_file(filename):
    cdef vector[vector[simplex]] result = process_file(filename.encode('utf-8'))
    return [[list(s.vertices) for s in simplex_list] for simplex_list in result]

def py_find_generators(chains):
    print(f"input")
    cdef vector[chain] c_chains = pylist_to_vector_chain(chains)
    print(f"cch")
    cdef vector[simplex] generators = find_generators(c_chains)
    print(f"GENs")
    return [list(g.vertices) for g in generators]

def py_create_matrix(generators, boundaries):
    return MatrixWrapper(generators, boundaries)
