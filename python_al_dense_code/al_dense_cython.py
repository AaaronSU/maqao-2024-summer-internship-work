import al_dense_cython_lib as al_dense_lib
import random

if __name__ == "__main__":
    vec_a = [1.0 for _ in range(20_000_000)]
    vec_b = [1.0 for _ in range(20_000_000)]
    sca   = 2.1
    # a = [random.uniform(0,1) for _ in range(100_000_000)]
    # b = [random.uniform(0,1) for _ in range(100_000_000)]
    vec_add_res      = al_dense_lib.vector_addition(vec_a, vec_b)
    vec_sca_mul_res  = al_dense_lib.vector_scalar_multiply(vec_a, sca)
    dot_pro_res      = al_dense_lib.dot_product(vec_a, vec_b)
    hadamard_pro_res = al_dense_lib.hadamard_product(vec_a, vec_b)
    saxpy_res        = al_dense_lib.saxpy(sca, vec_a, vec_b)
    print("vector addition result:             ", vec_add_res[:5])
    print("vector scalar multiplication result:", vec_sca_mul_res[:5])
    print("vector dot product result :         ", dot_pro_res)

    matrix_a = [[1.0 for _ in range(500)] for _ in range(1_000)] # matrix sized 1000 x 600
    matrix_b = [[1.0 for _ in range(1_000)] for _ in range(500)] # matrix sized 600 x 1000
    vec_c    = [1.0 for _ in range(500)]

    matrix_b_trans_res  = al_dense_lib.matrix_transpose(matrix_b)
    matrix_add_res      = al_dense_lib.matrix_addition(matrix_a, matrix_b_trans_res)
    matrix_sca_mul_res  = al_dense_lib.matrix_scalar_multiply(matrix_a, sca)
    matrix_vec_mul_res  = al_dense_lib.matrix_vector_multiply(matrix_a, vec_c)
    matrix_multiply_res = al_dense_lib.matrix_multiply(matrix_a, matrix_b)

    print("matrix transpose result:            ", matrix_b_trans_res[0][:5])
    print("matrix addition result:             ", matrix_add_res[0][:5])
    print("matrix scalar multiplication result:", matrix_sca_mul_res[0][:5])
    print("matrix vector multiplication result:", matrix_vec_mul_res[:5])
    print("matrix multiply result:             ", matrix_multiply_res[0][:5])

