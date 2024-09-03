cdef class CustomError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def vector_scalar_multiply(list A, float B):
    cdef int n = len(A)
    cdef list result = [0] * n
    cdef int i
    for i in range(n):
        result[i] = A[i] * B
    return result

def vector_addition(list A, list B):
    cdef int n = len(A)
    if n != len(B):
        raise CustomError("vector_addition: vector size doesn't match")
    cdef list result = [0] * n
    cdef int i
    for i in range(n):
        result[i] = A[i] + B[i]
    return result

def dot_product(list A, list B):
    cdef int n = len(A)
    if n != len(B):
        raise CustomError("dot_product: vector size doesn't match")
    cdef float sum = 0
    cdef int i
    for i in range(n):
        sum += A[i] * B[i]
    return sum

def hadamard_product(list A, list B):
    cdef int n = len(A)
    if n != len(B):
        raise CustomError("hadamard_product: vector size doesn't match")
    cdef list result = [0] * n
    cdef int i
    for i in range(n):
        result[i] = A[i] * B[i]
    return result

def saxpy(float a, list X, list Y):
    cdef int n = len(X)
    if n != len(Y):
        raise CustomError("saxpy: vector X and Y size doesn't match")
    cdef list result = [0] * n
    cdef int i
    for i in range(n):
        result[i] = a * X[i] + Y[i]
    return result

def matrix_transpose(list[list] A):
    cdef int rows = len(A)
    cdef int cols = len(A[0])
    cdef int i, j
    for i in range(rows - 1):
        if len(A[i]) != len(A[i + 1]):
            raise CustomError("matrix_transpose: matrix row size not equal")
    cdef list[list] B = [[0] * rows for _ in range(cols)]

    for i in range(rows):
        for j in range(len(A[i])):
            B[j][i] = A[i][j]
    return B

def matrix_addition(list[list] A, list[list] B):
    cdef int A_row_size = len(A)
    cdef int B_row_size = len(B)
    cdef int i, j
    if A_row_size != B_row_size:
        raise CustomError("matrix_addition: matrix size doesn't match")
    for i in range(A_row_size - 1):
        if len(A[i]) != len(A[i + 1]) or len(B[i]) != len(B[i + 1]) or len(A[i]) != len(B[i]):
            raise CustomError("matrix_addition: matrix size doesn't match")
    if len(A[A_row_size - 1]) != len(B[A_row_size - 1]):
        raise CustomError("matrix_addition: matrix size doesn't match")
    cdef list[list] C = [[0] * len(A[0]) for _ in range(A_row_size)]
    
    for i in range(A_row_size):
        for j in range(len(A[i])):
            C[i][j] = A[i][j] + B[i][j]
    return C

def matrix_scalar_multiply(list[list] A, float B):
    cdef int rows = len(A)
    cdef int cols = len(A[0])
    cdef int i, j
    for i in range(rows - 1):
        if len(A[i]) != len(A[i + 1]):
            raise CustomError("matrix_scalar_multiply: matrix row size not equal")
    cdef list[list] C = [[0] * cols for _ in range(rows)]
    
    for i in range(rows):
        for j in range(cols):
            C[i][j] = A[i][j] * B
    return C

def matrix_vector_multiply(list[list] A, list B):
    cdef int rows = len(A)
    cdef int cols = len(A[0])
    cdef int i, j

    for i in range(1, rows):
        if len(A[i]) != cols:
            raise CustomError("matrix_vector_multiply: matrix row size not equal")
    if cols != len(B):
        raise CustomError("matrix_vector_multiply: matrix row size doesn't match vector column size")
    cdef list C = [0] * rows
    
    for i in range(rows):
        for j in range(cols):
            C[i] += A[i][j] * B[j]
    return C

def matrix_multiply(list[list] A, list[list] B):
    cdef int A_rows = len(A)
    cdef int A_cols = len(A[0])
    cdef int B_rows = len(B)
    cdef int B_cols = len(B[0])
    cdef int i, j, k
    if A_cols != B_rows:
        raise CustomError("matrix_multiply: matrix A row size not equal to matrix B column size")
    for i in range(A_rows - 1):
        if len(A[i]) != len(A[i + 1]):
            raise CustomError("matrix_multiply: matrix A row size not equal")
    for i in range(B_rows - 1):
        if len(B[i]) != len(B[i + 1]):
            raise CustomError("matrix_multiply: matrix B row size not equal")
    cdef list[list] C = [[0] * B_cols for _ in range(A_rows)]

    for i in range(A_rows):
        for j in range(B_cols):
            for k in range(B_rows):
                C[i][j] += A[i][k] * B[k][j]
    return C