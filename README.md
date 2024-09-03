This repository contains the work completed during my summer internship 2024 for MAQAO. To use it, you must comply with the following License: [LGPL2.1](https://gitlab.liparad.uvsq.fr/maqao_release/MAQAO/-/blob/master/LGPL2.1)/[3.0](https://gitlab.liparad.uvsq.fr/maqao_release/MAQAO/-/blob/master/LGPL3.0).

The Titanic dataset stored in the `data` directory is sourced from the [Kaggle](https://www.kaggle.com/) website. If there are any licensing concerns, please contact me to request its removal.

The code is divided into two sections: artificial intelligence code (`ai_code`) and algebra dense code (`al_code`).

## Artificial Intelligence Code

The `ai_code` directory contains three programs that use different libraries:

- Titanic dataset analysis using the `scikit-learn` package.
- MNIST dataset analysis using the `TensorFlow` package.
- MNIST dataset analysis using the `PyTorch` package.

## Algebra Dense Code

The `al_code` directory contains:

- Three implementations of general linear algebra operations using different methods (pure Python, Cython, and C extension). These operations include vector addition, vector-scalar multiplication, dot product, Hadamard product, SAXPY, matrix transpose, matrix addition, matrix-scalar multiplication, matrix-vector multiplication, and matrix multiplication.
- A pure Python implementation of Jacobi and Gauss-Seidel iterations, which is not actively used.

## Summer Internship Objective

The goal of the summer internship was to explore and understand the capabilities and limitations of MAQAO on artificial intelligence code and on multiple Python compilers across different versions.

## Requirements

**Storage Requirement:** At least 128 GB of storage is needed for the experiments.

To reproduce the experiments, the following Python interpreters are required:

1. **PyPy 3.9.18**
2. **CPython 3.9.19**
3. **CPython 3.12.0**
4. **CPython 3.12.0** compiled with `-march=native` and with `-fno-omit-frame-pointer` (referred to as **CPython 3.12.0 opti** in the tests)
5. **IntelPython 3.9.19**
6. **Stackless 3.7.5**

## Installation Instructions

- **IntelPython** should be installed from the official website.
- **PyPy 3.9.18**, **CPython 3.9.19**, and **CPython 3.12.0** can be installed using Conda.
- **CPython 3.12.0 opti** and **Stackless** can be installed using Pyenv.

To install IntelPython, refer to the official installation tutorial.

For **Conda**, use the following command:
```bash
conda create -n your_env_name your_package
```

For Pyenv, use the following command:
```bash
pyenv install your_package
```

## Setup Instructions

1. After installing the compilers, find the absolute path where each interpreter is installed.

2. Use the command below to install the required packages, replacing python with the absolute path of the interpreter:

```bash
python -m pip install -r requirements.txt
```

Due to version incompatibility, use `requirements_stackless.txt` instead of `requirements.txt`git for Stackless 3.7.5.

3. Install Cython for the Python interpreters to test the algebra dense code using:

```bash
python -m pip install Cython
```
(Replace python with the specific interpreter you want to test.)

4. Navigate to the configurations_files directory and modify the executable names with their absolute paths.

5. Finally, generate the reports using Make. It takes around 40 minutes to generate all the reports.