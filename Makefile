# include definitions.mk
#####################################################################

PYTHON_3.9.19_PATH      := ~/.conda/envs/python3.9.19-env/bin/python
PYTHON_3.12.0_PATH      := ~/.conda/envs/python3.12.0-env/bin/python
PYTHON_3.12.0_OPTI_PATH := ~/.pyenv/versions/3.12.0/bin/python
INTELPYTHON_3.9.19_PATH := ~/intelpython3/bin/python
PYPY_3.9.18_PATH        := ~/.conda/envs/pypy-env/bin/python

#####################################################################

PYTHON_COMPILER      := python3.12.0_opti
PYTHON_COMPILER_PATH := $(PYTHON_3.12.0_OPTI_PATH)
NODE                 := 6
REPORT_DIR           := python_ai_report_dir

MAQAO_OPTIONS        := lprof_params=--include-kernel keep_executable_location=true



all: al_dense_setup
	./configurations/gen_report.sh

al_dense:
	cd python_al_dense_code && make setup && make test && make gen_report && make clean && cd ..

