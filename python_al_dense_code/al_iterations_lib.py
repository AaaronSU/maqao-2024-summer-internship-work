import numpy as np

def generate_strictly_diagonally_dominant_matrix(n, min_val=-1, max_val=1):
    A = np.random.uniform(min_val, max_val, size=(n, n)).astype(float)
    
    # Make the matrix strictly diagonally dominant
    for i in range(n):
        row_sum = sum(abs(A[i, j]) for j in range(n) if i != j)
        A[i, i] = row_sum + np.random.uniform(0, max_val)
    
    return A

def verify_strictly_diagonally_dominant(matrix):
    n = len(matrix)
    for i in range(n):
        diagonal_element = abs(matrix[i, i])
        off_diagonal_sum = sum(abs(matrix[i, j]) for j in range(n) if i != j)
        if diagonal_element <= off_diagonal_sum:
            return False
    return True


def jacobi(A, b, x0, tolerance, max_iterations):
    n = len(b)
    x = x0.copy()
    x_new = np.zeros_like(x) # initialize a vector with the same size of x
    
    for it in range(max_iterations):
        for i in range(n):
            s        = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        
        # Verify if the difference are less than the tolerance
        if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
            return x_new, it
        
        x = x_new.copy()
    
    raise ValueError("Jacobi method did not converge")

def gauss_seidel(A, b, x0, tolerance, max_iterations):
    n = len(b)
    x = x0.copy()
    
    for it in range(max_iterations):
        x_new = x.copy()
        for i in range(n):
            s1       = sum(A[i][j] * x_new[j] for j in range(i))
            s2       = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        
        # Verify if the difference are less than the tolerance
        if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
            return x_new, it
        
        x = x_new
    
    raise ValueError("Gauss-Seidel method did not converge")
