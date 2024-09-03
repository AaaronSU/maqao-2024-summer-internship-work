#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject *CustomError;

static PyObject* vector_scalar_multiply(PyObject* self, PyObject* args) {
    PyObject* list_obj;
    double scalar;
    if (!PyArg_ParseTuple(args, "Od", &list_obj, &scalar)) { // "Od" stand for Object double
        return NULL;
    }
    if (!PyList_Check(list_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list");
        return NULL;
    }
    
    Py_ssize_t list_size = PyList_Size(list_obj);
    PyObject* result_list = PyList_New(list_size);

    for (Py_ssize_t i = 0; i < list_size; i++) {
        PyObject* item = PyList_GetItem(list_obj, i);
        double value = PyFloat_AsDouble(item);
        PyObject* result_item = PyFloat_FromDouble(value * scalar);
        PyList_SetItem(result_list, i, result_item);
    }
    
    return result_list;
}

static PyObject* vector_addition(PyObject* self, PyObject* args) {
    PyObject* list1_obj;
    PyObject* list2_obj;
    if (!PyArg_ParseTuple(args, "OO", &list1_obj, &list2_obj)) {
        return NULL;
    }
    if (!PyList_Check(list1_obj) || !PyList_Check(list2_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected two lists");
        return NULL;
    }

    Py_ssize_t list_size = PyList_Size(list1_obj);
    if (list_size != PyList_Size(list2_obj)) {
        PyErr_SetString(CustomError, "vector_addition: vector size doesn't match");
        return NULL;
    }

    PyObject* result_list = PyList_New(list_size);

    for (Py_ssize_t i = 0; i < list_size; i++) {
        PyObject* item1 = PyList_GetItem(list1_obj, i);
        PyObject* item2 = PyList_GetItem(list2_obj, i);
        double value1 = PyFloat_AsDouble(item1);
        double value2 = PyFloat_AsDouble(item2);
        PyObject* result_item = PyFloat_FromDouble(value1 + value2);
        PyList_SetItem(result_list, i, result_item);
    }
    
    return result_list;
}

static PyObject* dot_product(PyObject* self, PyObject* args) {
    PyObject* list1_obj;
    PyObject* list2_obj;
    if (!PyArg_ParseTuple(args, "OO", &list1_obj, &list2_obj)) {
        return NULL;
    }
    if (!PyList_Check(list1_obj) || !PyList_Check(list2_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected two lists");
        return NULL;
    }

    Py_ssize_t list_size = PyList_Size(list1_obj);
    if (list_size != PyList_Size(list2_obj)) {
        PyErr_SetString(CustomError, "dot_product: vector size doesn't match");
        return NULL;
    }

    double dot = 0.0;
    for (Py_ssize_t i = 0; i < list_size; i++) {
        PyObject* item1 = PyList_GetItem(list1_obj, i);
        PyObject* item2 = PyList_GetItem(list2_obj, i);
        double value1 = PyFloat_AsDouble(item1);
        double value2 = PyFloat_AsDouble(item2);
        dot += value1 * value2;
    }
    
    return Py_BuildValue("d", dot);
}

static PyObject* hadamard_product(PyObject* self, PyObject* args) {
    PyObject* list1_obj;
    PyObject* list2_obj;
    if (!PyArg_ParseTuple(args, "OO", &list1_obj, &list2_obj)) {
        return NULL;
    }
    if (!PyList_Check(list1_obj) || !PyList_Check(list2_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected two lists");
        return NULL;
    }

    Py_ssize_t list_size = PyList_Size(list1_obj);
    if (list_size != PyList_Size(list2_obj)) {
        PyErr_SetString(CustomError, "hadamard_product: vector size doesn't match");
        return NULL;
    }

    PyObject* result_list = PyList_New(list_size);

    for (Py_ssize_t i = 0; i < list_size; i++) {
        PyObject* item1 = PyList_GetItem(list1_obj, i);
        PyObject* item2 = PyList_GetItem(list2_obj, i);
        double value1 = PyFloat_AsDouble(item1);
        double value2 = PyFloat_AsDouble(item2);
        PyObject* result_item = PyFloat_FromDouble(value1 * value2);
        PyList_SetItem(result_list, i, result_item);
    }
    
    return result_list;
}

// SAXPY (y = alpha * x + y)
static PyObject* saxpy(PyObject* self, PyObject* args) {
    double alpha;
    PyObject* x_list_obj;
    PyObject* y_list_obj;
    if (!PyArg_ParseTuple(args, "dOO", &alpha, &x_list_obj, &y_list_obj)) {
        return NULL;
    }
    if (!PyList_Check(x_list_obj) || !PyList_Check(y_list_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected two lists");
        return NULL;
    }

    Py_ssize_t list_size = PyList_Size(x_list_obj);
    if (list_size != PyList_Size(y_list_obj)) {
        PyErr_SetString(CustomError, "saxpy: vector X and Y size doesn't match");
        return NULL;
    }

    PyObject* result_list = PyList_New(list_size);

    for (Py_ssize_t i = 0; i < list_size; i++) {
        PyObject* x_item = PyList_GetItem(x_list_obj, i);
        PyObject* y_item = PyList_GetItem(y_list_obj, i);
        double x_value = PyFloat_AsDouble(x_item);
        double y_value = PyFloat_AsDouble(y_item);
        PyObject* result_item = PyFloat_FromDouble(alpha * x_value + y_value);
        PyList_SetItem(result_list, i, result_item);
    }
    
    return result_list;
}

static PyObject* matrix_transpose(PyObject* self, PyObject* args) {
    PyObject* matrix_obj;
    if (!PyArg_ParseTuple(args, "O", &matrix_obj)) {
        return NULL;
    }
    if (!PyList_Check(matrix_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list of lists");
        return NULL;
    }

    Py_ssize_t rows = PyList_Size(matrix_obj);
    if (rows == 0) {
        PyErr_SetString(PyExc_ValueError, "Matrix cannot be empty");
        return NULL;
    }
    Py_ssize_t cols = PyList_Size(PyList_GetItem(matrix_obj, 0));
    
    for (Py_ssize_t i = 1; i < rows; i++) {
        PyObject* row = PyList_GetItem(matrix_obj, i);
        if (!PyList_Check(row) || PyList_Size(row) != cols) {
            PyErr_SetString(CustomError, "matrix_transpose: matrix row size not equal");
            return NULL;
        }
    }

    PyObject* result_matrix = PyList_New(cols);
    for (Py_ssize_t j = 0; j < cols; j++) {
        PyObject* new_row = PyList_New(rows);
        for (Py_ssize_t i = 0; i < rows; i++) {
            PyObject* item = PyList_GetItem(PyList_GetItem(matrix_obj, i), j);
            Py_INCREF(item);
            PyList_SetItem(new_row, i, item);
        }
        PyList_SetItem(result_matrix, j, new_row);
    }

    return result_matrix;
}

static PyObject* matrix_addition(PyObject* self, PyObject* args) {
    PyObject *A_obj, *B_obj;
    if (!PyArg_ParseTuple(args, "OO", &A_obj, &B_obj)) {
        PyErr_SetString(CustomError, "Invalid arguments");
        return NULL;
    }

    if (!PyList_Check(A_obj) || !PyList_Check(B_obj)) {
        PyErr_SetString(CustomError, "Arguments must be lists");
        return NULL;
    }

    Py_ssize_t A_row_size = PyList_Size(A_obj);
    Py_ssize_t B_row_size = PyList_Size(B_obj);

    if (A_row_size != B_row_size) {
        PyErr_SetString(CustomError, "matrix_addition: matrix size doesn't match");
        return NULL;
    }

    // Create the result matrix
    PyObject *C_obj = PyList_New(A_row_size);
    if (C_obj == NULL) {
        return NULL;
    }

    for (Py_ssize_t i = 0; i < A_row_size; i++) {
        PyObject *A_row = PyList_GetItem(A_obj, i);
        PyObject *B_row = PyList_GetItem(B_obj, i);

        if (!PyList_Check(A_row) || !PyList_Check(B_row) ||
            PyList_Size(A_row) != PyList_Size(B_row)) {
            PyErr_SetString(CustomError, "matrix_addition: matrix size doesn't match");
            Py_DECREF(C_obj);
            return NULL;
        }

        PyObject *C_row = PyList_New(PyList_Size(A_row));
        if (C_row == NULL) {
            Py_DECREF(C_obj);
            return NULL;
        }

        for (Py_ssize_t j = 0; j < PyList_Size(A_row); j++) {
            PyObject *A_elem = PyList_GetItem(A_row, j);
            PyObject *B_elem = PyList_GetItem(B_row, j);

            PyObject *C_elem = PyFloat_FromDouble(PyFloat_AsDouble(A_elem) + PyFloat_AsDouble(B_elem));
            if (C_elem == NULL) {
                Py_DECREF(C_row);
                Py_DECREF(C_obj);
                return NULL;
            }

            PyList_SetItem(C_row, j, C_elem);
        }

        PyList_SetItem(C_obj, i, C_row);
    }

    return C_obj;
}

static PyObject* matrix_scalar_multiply(PyObject* self, PyObject* args) {
    PyObject* matrix_obj;
    double scalar;
    if (!PyArg_ParseTuple(args, "Od", &matrix_obj, &scalar)) {
        return NULL;
    }
    if (!PyList_Check(matrix_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected a matrix (list of lists)");
        return NULL;
    }

    Py_ssize_t rows = PyList_Size(matrix_obj);
    Py_ssize_t cols = PyList_Size(PyList_GetItem(matrix_obj, 0));

    for (Py_ssize_t i = 1; i < rows; i++) {
        PyObject* row = PyList_GetItem(matrix_obj, i);
        if (!PyList_Check(row) || PyList_Size(row) != cols) {
            PyErr_SetString(CustomError, "matrix_scalar_multiply: matrix row size not equal");
            return NULL;
        }
    }

    PyObject* result_matrix = PyList_New(rows);
    for (Py_ssize_t i = 0; i < rows; i++) {
        PyObject* new_row = PyList_New(cols);
        for (Py_ssize_t j = 0; j < cols; j++) {
            double value = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(matrix_obj, i), j));
            PyObject* result_item = PyFloat_FromDouble(value * scalar);
            PyList_SetItem(new_row, j, result_item);
        }
        PyList_SetItem(result_matrix, i, new_row);
    }

    return result_matrix;
}

static PyObject* matrix_vector_multiply(PyObject* self, PyObject* args) {
    PyObject *matrix_obj, *vector_obj;
    PyObject *row_obj;
    Py_ssize_t i, j, num_rows, num_cols;

    if (!PyArg_ParseTuple(args, "OO", &matrix_obj, &vector_obj)) {
        return NULL;
    }

    if (!PyList_Check(matrix_obj) || !PyList_Check(vector_obj)) {
        PyErr_SetString(PyExc_TypeError, "Arguments must be lists");
        return NULL;
    }

    num_rows = PyList_Size(matrix_obj);
    if (num_rows == 0) {
        PyErr_SetString(PyExc_ValueError, "Matrix must have at least one row");
        return NULL;
    }

    row_obj = PyList_GetItem(matrix_obj, 0);
    if (!PyList_Check(row_obj)) {
        PyErr_SetString(PyExc_ValueError, "First row of the matrix must be a list");
        return NULL;
    }

    num_cols = PyList_Size(row_obj);

    for (Py_ssize_t i = 1; i < num_rows; i++) {
        PyObject* row = PyList_GetItem(matrix_obj, i);
        if (!PyList_Check(row) || PyList_Size(row) != num_cols) {
            PyErr_SetString(CustomError, "matrix_vector_multiply: matrix row size not equal");
            return NULL;
        }
    }

    if (num_cols != PyList_Size(vector_obj)) {
        PyErr_SetString(CustomError, "matrix_vector_multiply: matrix row size doesn't match vector column size");
        return NULL;
    }

    PyObject *result_list = PyList_New(num_rows);
    if (result_list == NULL) {
        return NULL;
    }

    // matrix-vector multiplication
    for (i = 0; i < num_rows; i++) {
        PyObject *row = PyList_GetItem(matrix_obj, i);
        double sum = 0.0;

        for (j = 0; j < num_cols; j++) {
            PyObject *matrix_element = PyList_GetItem(row, j);
            PyObject *vector_element = PyList_GetItem(vector_obj, j);
            double matrix_value = PyFloat_AsDouble(matrix_element);
            double vector_value = PyFloat_AsDouble(vector_element);
            sum += matrix_value * vector_value;
        }

        PyObject *result_value = PyFloat_FromDouble(sum);
        PyList_SetItem(result_list, i, result_value);
    }

    return result_list;
}

static PyObject* matrix_multiply(PyObject* self, PyObject* args) {
    PyObject *A_obj, *B_obj;
    if (!PyArg_ParseTuple(args, "OO", &A_obj, &B_obj)) {
        PyErr_SetString(CustomError, "Invalid arguments");
        return NULL;
    }

    if (!PyList_Check(A_obj) || !PyList_Check(B_obj)) {
        PyErr_SetString(CustomError, "Arguments must be lists");
        return NULL;
    }

    Py_ssize_t A_row_size = PyList_Size(A_obj);
    Py_ssize_t B_row_size = PyList_Size(B_obj);

    if (A_row_size == 0 || B_row_size == 0) {
        PyErr_SetString(CustomError, "Matrices cannot be empty");
        return NULL;
    }

    PyObject *A_first_row = PyList_GetItem(A_obj, 0);
    PyObject *B_first_row = PyList_GetItem(B_obj, 0);

    if (!PyList_Check(A_first_row) || !PyList_Check(B_first_row)) {
        PyErr_SetString(CustomError, "Each matrix must be a list of lists");
        return NULL;
    }

    Py_ssize_t A_col_size = PyList_Size(A_first_row);
    Py_ssize_t B_col_size = PyList_Size(B_first_row);

    for (Py_ssize_t i = 1; i < A_row_size; i++) {
        PyObject *row = PyList_GetItem(A_obj, i);
        if (!PyList_Check(row) || PyList_Size(row) != A_col_size) {
            PyErr_SetString(CustomError, "matrix_multiply: matrix A row size not equal");
            return NULL;
        }
    }

    for (Py_ssize_t i = 1; i < B_row_size; i++) {
        PyObject *row = PyList_GetItem(B_obj, i);
        if (!PyList_Check(row) || PyList_Size(row) != B_col_size) {
            PyErr_SetString(CustomError, "matrix_multiply: matrix B row size not equal");
            return NULL;
        }
    }

    if (A_col_size != B_row_size) {
        PyErr_SetString(CustomError, "matrix_multiply: matrix A row size not equal to matrix B column size");
        return NULL;
    }

    // result C 
    PyObject *C_obj = PyList_New(A_row_size);
    if (C_obj == NULL) {
        return NULL;
    }

    for (Py_ssize_t i = 0; i < A_row_size; i++) {
        PyObject *A_row = PyList_GetItem(A_obj, i);
        if (!PyList_Check(A_row)) {
            PyErr_SetString(CustomError, "Matrix rows must be lists");
            Py_DECREF(C_obj);
            return NULL;
        }

        PyObject *C_row = PyList_New(B_col_size);
        if (C_row == NULL) {
            Py_DECREF(C_obj);
            return NULL;
        }

        for (Py_ssize_t j = 0; j < B_col_size; j++) {
            double sum = 0.0;

            for (Py_ssize_t k = 0; k < A_col_size; k++) {
                PyObject *A_elem = PyList_GetItem(A_row, k);
                PyObject *B_elem = PyList_GetItem(PyList_GetItem(B_obj, k), j);

                sum += PyFloat_AsDouble(A_elem) * PyFloat_AsDouble(B_elem);
            }

            PyObject *C_elem = PyFloat_FromDouble(sum);
            if (C_elem == NULL) {
                Py_DECREF(C_row);
                Py_DECREF(C_obj);
                return NULL;
            }

            PyList_SetItem(C_row, j, C_elem);
        }

        PyList_SetItem(C_obj, i, C_row);
    }

    return C_obj;
}

// Define the method table
static PyMethodDef LibMethods[] = {
    {"vector_scalar_multiply", vector_scalar_multiply, METH_VARARGS, "Multiply a vector by a scalar"},
    {"vector_addition",             vector_addition,             METH_VARARGS, "Add two vectors"},
    {"dot_product",            dot_product,            METH_VARARGS, "Compute the dot product of two vectors"},
    {"hadamard_product",       hadamard_product,       METH_VARARGS, "Compute the Hadamard product of two vectors"},
    {"saxpy",                  saxpy,                  METH_VARARGS, "Compute SAXPY (y = alpha * x + y)"},
    {"matrix_transpose",       matrix_transpose,       METH_VARARGS, "Transpose a matrix"},
    {"matrix_addition",        matrix_addition,        METH_VARARGS, "Add two matrices"},
    {"matrix_scalar_multiply", matrix_scalar_multiply, METH_VARARGS, "Multiply a matrix by a scalar"},
    {"matrix_vector_multiply", matrix_vector_multiply, METH_VARARGS, "Multiply a matrix by a scalar"},
    {"matrix_multiply",        matrix_multiply,        METH_VARARGS, "Multiply two matrices"},
    {NULL, NULL, 0, NULL}
};

// Define the module
static struct PyModuleDef al_dense_lib = {
    PyModuleDef_HEAD_INIT,
    "c_extension_al_dense_lib",
    NULL,
    -1,
    LibMethods
};

// Initialize the module
PyMODINIT_FUNC PyInit_al_dense_c_extension_lib(void) {
    PyObject *module;

    module = PyModule_Create(&al_dense_lib);
    if (module == NULL)
        return NULL;

    /* Create the custom exception */
    CustomError = PyErr_NewException("matrix_vector.CustomError", NULL, NULL);
    if (CustomError == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    /* Add the exception to the module */
    PyModule_AddObject(module, "CustomError", CustomError);
    return module;
}
