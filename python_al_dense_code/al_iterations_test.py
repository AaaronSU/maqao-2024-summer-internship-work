class TestAlgebraIteration(unittest.TestCase):
    def test_generate_strictly_diagonally_dominant_matrix(self):
        n = 5
        A = al_iterations_lib.generate_strictly_diagonally_dominant_matrix(n)
        self.assertTrue(al_iterations_lib.verify_strictly_diagonally_dominant(A), "Matrix is not strictly diagonally dominant")
    
    def test_jacobi(self):
        n = 100
        A = al_iterations_lib.generate_strictly_diagonally_dominant_matrix(n)
        b = np.random.rand(n)
        x0 = np.zeros(n)
        tol = 1e-5
        max_iterations = 1000
        x, iters = al_iterations_lib.jacobi(A, b, x0, tol, max_iterations)
        # Verify if A*x = b
        np.testing.assert_allclose(np.dot(A, x), b, rtol=tol, atol=tol*10)
    
    def test_gauss_seidel(self):
        n = 4
        A = al_iterations_lib.generate_strictly_diagonally_dominant_matrix(n)
        b = np.random.rand(n)
        x0 = np.zeros(n)
        tol = 1e-5
        max_iterations = 1000
        x, iters = al_iterations_lib.gauss_seidel(A, b, x0, tol, max_iterations)
        # Verify if A*x = b
        np.testing.assert_allclose(np.dot(A, x), b, rtol=tol, atol=tol*2)


if __name__ == "__main__":
    unittest.main()
