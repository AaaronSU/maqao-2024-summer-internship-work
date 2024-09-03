.PHONY: all setup test clean

all: setup test

setup:
	$(PYTHON_COMPILER) setup.py build_ext --inplace
	$(PYTHON_COMPILER) setup.py clean

test:
	$(PYTHON_COMPILER) al_test_c_extension.py
	$(PYTHON_COMPILER) al_test_cython.py
	$(PYTHON_COMPILER) al_test.py

clean:
	rm *.so al_dense_cython_lib.c