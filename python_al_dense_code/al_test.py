import al_dense_lib
import unittest
from al_dense_lib import CustomError

class TestAlgebraDense(unittest.TestCase):

    def test_vector_scalar_multiply(self):
        A = [1,2,3,4]
        B = 5
        C = [5,10,15,20]
        self.assertEqual(al_dense_lib.vector_scalar_multiply(A, B), C)

    def test_vector_addition(self):
        A = [1,2,3,4]
        B = [5,6,7,8]
        C = [6,8,10,12]
        self.assertEqual(al_dense_lib.vector_addition(A, B), C)

        A = [1,2,3]
        B = [5,6,7,8]
        with self.assertRaises(CustomError) as context:
            al_dense_lib.vector_addition(A, B)
        self.assertEqual(str(context.exception), "vector_addition: vector size doesn't match")

    def test_dot_product(self):
        A = [1,2,3,4]
        B = [5,6,7,8]
        C = 70
        self.assertEqual(al_dense_lib.dot_product(A, B), C)

        A = [1,2,3]
        B = [5,6,7,8]
        with self.assertRaises(CustomError) as context:
            al_dense_lib.dot_product(A, B)
        self.assertEqual(str(context.exception), "dot_product: vector size doesn't match")

    def test_hadamard_product(self):
        A = [1,2,3,4]
        B = [5,6,7,8]
        C = [5,12,21,32]
        self.assertEqual(al_dense_lib.hadamard_product(A, B), C)

        A = [1,2,3]
        B = [5,6,7,8]
        with self.assertRaises(CustomError) as context:
            al_dense_lib.hadamard_product(A, B)
        self.assertEqual(str(context.exception), "hadamard_product: vector size doesn't match")

    def test_saxpy(self):
        a = 5
        X = [1, 2, 3]
        Y = [5, 6, 7]
        result = [10, 16, 22]
        self.assertEqual(al_dense_lib.saxpy(a, X, Y), result)
        
        a = 5
        X = [1, 2, 3]
        Y = [5, 6]
        with self.assertRaises(CustomError) as context:
            al_dense_lib.saxpy(a, X, Y)
        self.assertEqual(str(context.exception), "saxpy: vector X and Y size doesn't match")

    def test_matrix_transpose(self):
        A = [
            [1,2,3],
            [4,5],
            [7,8,9]
        ]
        with self.assertRaises(CustomError) as context:
            al_dense_lib.matrix_transpose(A)
        self.assertEqual(str(context.exception), "matrix_transpose: matrix row size not equal")

        A = [[1, 2, 3, 4]]
        B = [[1], [2], [3], [4]]
        self.assertEqual(al_dense_lib.matrix_transpose(A), B)
        self.assertEqual(al_dense_lib.matrix_transpose(B), A)

    def test_matrix_addition(self):
        A = [
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ]
        B = [
            [7,7,7],
            [6,8,9],
            [2,6,8]
        ]
        result = [
            [8,9,10],
            [10,13,15],
            [9,14,17]
        ]
        self.assertEqual(al_dense_lib.matrix_addition(A, B), result)
        

        A = [
            [1,2,3],
            [4,5],
            [7,8,9]
        ]
        B = [
            [7,7,7],
            [6,8,9],
            [2,6,8]
        ]
        with self.assertRaises(CustomError) as context:
            al_dense_lib.matrix_addition(A,B)
        self.assertEqual(str(context.exception), "matrix_addition: matrix size doesn't match")

        A = [
            [1,2,3],
            [4,5,6],
        ]
        B = [
            [7,7,7],
            [6,8,9],
            [2,6,8]
        ]
        with self.assertRaises(CustomError) as context:
            al_dense_lib.matrix_addition(A, B)
        self.assertEqual(str(context.exception), "matrix_addition: matrix size doesn't match")

        A = [
            [1,2,3],
            [4,5,6],
        ]
        B = [
            [7,7],
            [6,8,9],
            [2,6,8]
        ]
        with self.assertRaises(CustomError) as context:
            al_dense_lib.matrix_addition(A, B)
        self.assertEqual(str(context.exception), "matrix_addition: matrix size doesn't match")

    def test_matrix_scalar_multiply(self):
        A = [
            [1,2,3],
            [4,5,8],
            [7,8,9]
        ]
        B = 5
        C = [
            [5,10,15],
            [20,25,40],
            [35,40,45]
        ]
        self.assertEqual(al_dense_lib.matrix_scalar_multiply(A, B), C)

        A = [
            [1,2],
            [4,5,8],
            [7,8,9]
        ]
        B = 5
        with self.assertRaises(CustomError) as context:
            al_dense_lib.matrix_scalar_multiply(A, B)
        self.assertEqual(str(context.exception), "matrix_scalar_multiply: matrix row size not equal")

    def test_matrix_vector_multiply(self):
        A = [
            [1,2,3],
            [4,5,8],
            [7,8,9]
        ]
        B = [1,2,3]
        C = [14,38,50]
        self.assertEqual(al_dense_lib.matrix_vector_multiply(A, B), C)

        A = [
            [1,2],
            [4,5,8],
            [7,8,9]
        ]
        B = [1,2,3]
        with self.assertRaises(CustomError) as context:
            al_dense_lib.matrix_vector_multiply(A, B)
        self.assertEqual(str(context.exception), "matrix_vector_multiply: matrix row size not equal")

        A = [
            [1,2,3],
            [4,5,8],
            [7,8,9]
        ]
        B = [1,2]
        with self.assertRaises(CustomError) as context:
            al_dense_lib.matrix_vector_multiply(A, B)
        self.assertEqual(str(context.exception), "matrix_vector_multiply: matrix row size doesn't match vector column size")

    def test_matrix_multiply(self):
        A = [
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ]
        B = [
            [7,7,7],
            [6,8,9],
            [2,6,8]
        ]
        result = [
            [25, 41, 49], 
            [70, 104, 121], 
            [115, 167, 193]
        ]
        self.assertEqual(al_dense_lib.matrix_multiply(A, B), result)

        A = [
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ]
        B = [
            [7,7],
            [6,8],
            [2,6]
        ]
        result = [
            [25, 41], 
            [70, 104], 
            [115, 167]
        ]
        self.assertEqual(al_dense_lib.matrix_multiply(A, B), result)

        A = [
            [1,2,3],
            [4,5,6],
        ]
        B = [
            [7,7,7],
            [6,8,9],
            [2,6,8]
        ]
        result = [
            [25, 41, 49], 
            [70, 104, 121]
        ]
        self.assertEqual(al_dense_lib.matrix_multiply(A, B), result)
        
        A = [
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ]
        B = [
            [7,7,7],
            [6,8,9]
        ]
        with self.assertRaises(CustomError) as context:
            al_dense_lib.matrix_multiply(A, B)
        self.assertEqual(str(context.exception), "matrix_multiply: matrix A row size not equal to matrix B column size")

        A = [
            [1,2,3],
            [4,5],
            [7,8,9]
        ]
        B = [
            [7,7,7],
            [6,8,9],
            [2,6,8]
        ]
        with self.assertRaises(CustomError) as context:
            al_dense_lib.matrix_multiply(A,B)
        self.assertEqual(str(context.exception), "matrix_multiply: matrix A row size not equal")

        A = [
            [1,2,3],
            [4,5,6],
        ]
        B = [
            [7,7],
            [6,8,9],
            [2,6,8]
        ]
        with self.assertRaises(CustomError) as context:
            al_dense_lib.matrix_multiply(A, B)
        self.assertEqual(str(context.exception), "matrix_multiply: matrix B row size not equal")

if __name__ == "__main__":
    unittest.main()
