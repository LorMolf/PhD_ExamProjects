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
    for item in py_list:
        c = chain()  # Use default constructor
        for elem in item:
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

# Python-accessible functions
def py_process_file(filename):
    cdef vector[vector[simplex]] result = process_file(filename.encode('utf-8'))
    return [[list(s.vertices) for s in simplex_list] for simplex_list in result]

def py_find_generators(chains):
    cdef vector[chain] c_chains = pylist_to_vector_chain(chains)
    cdef vector[simplex] generators = find_generators(c_chains)
    return [list(g.vertices) for g in generators]

def py_create_matrix(generators, boundaries):
    return MatrixWrapper(generators, boundaries)
