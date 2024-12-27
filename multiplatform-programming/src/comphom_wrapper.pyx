# cython: language_level=3

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
ctypedef unsigned long ulong

# Update simplex class and constructor
cdef extern from "simplex.h":
    cdef cppclass simplex:
        simplex()
        vector[ulong] vertices
        chain boundary()
        void print()
        bint operator==(simplex other)

cdef extern from "simplex.h":
    cdef cppclass chain_element:
        chain_element()
        long c
        simplex s

cdef extern from "simplex.h":
    cdef cppclass chain:
        chain()
        vector[chain_element] elements
        void print()

cdef extern from "matrix.h":
    cdef cppclass matrix:
        matrix()
        matrix(const matrix& other)
        matrix assign(vector[long]&, ulong, ulong)
        matrix nf_smith()
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
        s = simplex()
        s.vertices = pylist_to_vector_ulong(item)
        c_vector.push_back(s)
    return c_vector

cdef vector[chain] pylist_to_vector_chain(list py_list):
    cdef vector[chain] c_vector
    cdef chain c
    cdef chain_element ce
    for item in py_list:
        c = chain()
        elem = item
        ce = chain_element()
        ce.c = elem[0]
        ce.s.vertices = pylist_to_vector_ulong(elem[1])
        c.elements.push_back(ce)
        c_vector.push_back(c)
    return c_vector

# MatrixWrapper class
cdef class MatrixWrapper:
    cdef matrix* c_matrix

    def __cinit__(self, list generators=None, list boundaries=None):
        self.c_matrix = NULL
        if generators is not None and boundaries is not None:
            self._init_from_generators(generators, boundaries)

    def _init_from_generators(self, list generators, list boundaries):
        self.c_matrix = new matrix(create_matrix(
            pylist_to_vector_simplex(generators),
            pylist_to_vector_chain(boundaries)
        ))

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

    @staticmethod
    cdef create_from_matrix(matrix* m):
        cdef MatrixWrapper wrapper = MatrixWrapper(None, None)
        wrapper.c_matrix = m
        return wrapper

    cpdef nf_smith(self):
        cdef size_t rows
        cdef size_t cols
        cdef matrix snf_res
        cdef matrix* snf_matrix = NULL
        cdef MatrixWrapper wrapper

        if self.c_matrix == NULL:
            print("Null matrix pointer")
            return None

        try:
            rows = self.c_matrix.get_num_rows()
            cols = self.c_matrix.get_num_cols()
            if rows == 0 or cols == 0:
                print(f"Invalid matrix dimensions: {rows}x{cols}")
                return None
                
            snf_res = self.c_matrix.nf_smith()
            snf_matrix = new matrix(snf_res)
            wrapper = MatrixWrapper.create_from_matrix(snf_matrix)
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
    cdef vector[chain] c_chains = pylist_to_vector_chain(chains)
    cdef vector[simplex] generators = find_generators(c_chains)
    return [list(g.vertices) for g in generators]

def py_create_matrix(generators, boundaries):
    return MatrixWrapper(generators, boundaries)
