#!/usr/bin/python3
class CustomError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def vector_scalar_multiply(A, B):
    return [A[_] * B for _ in range(len(A))]

def vector_addition(A, B):
    if len(A) != len(B):
        raise CustomError("vector_addition: vector size doesn't match")
    assert len(A) == len(B)
    return [A[_] + B[_] for _ in range(len(A))]

def dot_product(A, B):
    if len(A) != len(B):
        raise CustomError("dot_product: vector size doesn't match")
    sum = 0
    for _ in range(len(A)):
        sum += A[_] * B[_]
    return sum

def hadamard_product(A, B):
    if len(A) != len(B):
        raise CustomError("hadamard_product: vector size doesn't match")
    return [A[_] * B[_] for _ in range(len(A))]

def saxpy(a, X, Y):
    if len(X) != len(Y):
        raise CustomError("saxpy: vector X and Y size doesn't match")
    return [a * X[_] + Y[_] for _ in range(len(X))]

def matrix_transpose(A):
    for i in range(len(A)-1):
        if len(A[i]) != len(A[i+1]):
            raise CustomError("matrix_transpose: matrix row size not equal")

    B = [[None for _ in range(len(A))]for _ in range(len(A[0]))]
    for i in range(len(A)):
        for j in range(len(A[i])):
            B[j][i] = A[i][j]
    return B

def matrix_addition(A, B):
    A_row_size = len(A)
    B_row_size = len(B)
    if A_row_size != B_row_size:
        raise CustomError("matrix_addition: matrix size doesn't match")
    for i in range(A_row_size-1):
        if len(A[i]) != len(A[i+1]) or len(B[i]) != len(B[i+1]) or len(A[i]) != len(B[i]):
            raise CustomError("matrix_addition: matrix size doesn't match")
    if len(A[A_row_size-1]) != len(B[A_row_size-1]):
        raise CustomError("matrix_addition: matrix size doesn't match")

    C = [[None for _ in range(len(A[0]))]for _ in range(A_row_size)]
    for i in range(A_row_size):
        for j in range(len(A[i])):
            C[i][j] = A[i][j] + B[i][j]
    return C

def matrix_scalar_multiply(A, B):
    for i in range(len(A)-1):
        if len(A[i]) != len(A[i+1]):
            raise CustomError("matrix_scalar_multiply: matrix row size not equal")
    C = [[None for _ in range(len(A[0]))]for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[i])):
            C[i][j] = A[i][j] * B
    return C

def matrix_vector_multiply(A, B):
    for i in range(len(A)-1):
        if len(A[i]) != len(A[i+1]):
            raise CustomError("matrix_vector_multiply: matrix row size not equal")

    if len(A[0]) != len(B):
        raise CustomError("matrix_vector_multiply: matrix row size doesn't match vector column size")

    C = [0 for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            C[i] += A[i][j] * B[j]
    return C

def matrix_multiply(A, B):
    for i in range(len(A)-1):
        if len(A[i]) != len(A[i+1]):
            raise CustomError("matrix_multiply: matrix A row size not equal")

    for i in range(len(B)-1):
        if len(B[i]) != len(B[i+1]):
            raise CustomError("matrix_multiply: matrix B row size not equal")

    if len(A[0]) != len(B):
            raise CustomError("matrix_multiply: matrix A row size not equal to matrix B column size")

    C = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
    return C