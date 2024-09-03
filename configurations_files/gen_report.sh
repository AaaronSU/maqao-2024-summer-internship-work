# maqao OV --create-config-template=seq && mv template_seq.json python_ai_intelpython3.9.19_tensorflow_mnist_config.json
# maqao OV --create-config-template=seq && mv template_seq.json python_ai_intelpython3.9.19_torch_mnist_config.json
# maqao OV --create-config-template=seq && mv template_seq.json python_ai_intelpython3.9.19_titanic_config.json

# maqao OV --create-config-template=seq && mv template_seq.json python_ai_pypy3.9.18_titanic_config.json

# maqao OV --create-config-template=seq && mv template_seq.json python_ai_python3.9.19_tensorflow_mnist_config.json
# maqao OV --create-config-template=seq && mv template_seq.json python_ai_python3.9.19_torch_mnist_config.json
# maqao OV --create-config-template=seq && mv template_seq.json python_ai_python3.9.19_titanic_config.json

# maqao OV --create-config-template=seq && mv template_seq.json python_ai_python3.12.0_tensorflow_mnist_config.json
# maqao OV --create-config-template=seq && mv template_seq.json python_ai_python3.12.0_torch_mnist_config.json
# maqao OV --create-config-template=seq && mv template_seq.json python_ai_python3.12.0_titanic_config.json

# maqao OV --create-config-template=seq && mv template_seq.json python_ai_python3.12.0_opti_tensorflow_mnist_config.json
# maqao OV --create-config-template=seq && mv template_seq.json python_ai_python3.12.0_opti_torch_mnist_config.json
# maqao OV --create-config-template=seq && mv template_seq.json python_ai_python3.12.0_opti_titanic_config.json

date

mkdir -p final_reports
mkdir -p final_reports/tensorflow_mnist_report
mkdir -p final_reports/pytorch_mnist_report
mkdir -p final_reports/titanic_report
mkdir -p final_reports/al_dense_report

taskset -c 3 maqao OV -R1 xp=final_reports/tensorflow_mnist_report/intelpython3.9.19 -dbg=1 --replace --config=./configurations_files/python_ai_intelpython3.9.19_tensorflow_mnist_config.json
taskset -c 3 maqao OV -R1 xp=final_reports/pytorch_mnist_report/intelpython3.9.19    -dbg=1 --replace --config=./configurations_files/python_ai_intelpython3.9.19_torch_mnist_config.json
taskset -c 3 maqao OV -R1 xp=final_reports/titanic_report/intelpython3.9.19          -dbg=1 --replace --config=./configurations_files/python_ai_intelpython3.9.19_titanic_config.json

taskset -c 3 maqao OV -R1 xp=final_reports/titanic_report/pypy3.9.18                 -dbg=1 --replace --config=./configurations_files/python_ai_pypy3.9.18_titanic_config.json

taskset -c 3 maqao OV -R1 xp=final_reports/tensorflow_mnist_report/python3.9.19      -dbg=1 --replace --config=./configurations_files/python_ai_python3.9.19_tensorflow_mnist_config.json
taskset -c 3 maqao OV -R1 xp=final_reports/pytorch_mnist_report/python3.9.19         -dbg=1 --replace --config=./configurations_files/python_ai_python3.9.19_torch_mnist_config.json
taskset -c 3 maqao OV -R1 xp=final_reports/titanic_report/python3.9.19               -dbg=1 --replace --config=./configurations_files/python_ai_python3.9.19_titanic_config.json

taskset -c 2 maqao OV -R1 xp=final_reports/tensorflow_mnist_report/python3.12.0      -dbg=1 --replace --config=./configurations_files/python_ai_python3.12.0_tensorflow_mnist_config.json
taskset -c 2 maqao OV -R1 xp=final_reports/pytorch_mnist_report/python3.12.0         -dbg=1 --replace --config=./configurations_files/python_ai_python3.12.0_torch_mnist_config.json
taskset -c 2 maqao OV -R1 xp=final_reports/titanic_report/python3.12.0               -dbg=1 --replace --config=./configurations_files/python_ai_python3.12.0_titanic_config.json

taskset -c 2 maqao OV -R1 xp=final_reports/tensorflow_mnist_report/python3.12.0_opti -dbg=1 --replace --config=./configurations_files/python_ai_python3.12.0_opti_tensorflow_mnist_config.json
taskset -c 2 maqao OV -R1 xp=final_reports/pytorch_mnist_report/python3.12.0_opti    -dbg=1 --replace --config=./configurations_files/python_ai_python3.12.0_opti_torch_mnist_config.json
taskset -c 2 maqao OV -R1 xp=final_reports/titanic_report/python3.12.0_opti          -dbg=1 --replace --config=./configurations_files/python_ai_python3.12.0_opti_titanic_config.json

taskset -c 2 maqao OV -R1 xp=final_reports/pytorch_mnist_report/stackless3.7.5       -dbg=1 --replace --config=./configurations_files/python_ai_stackless3.7.5_torch_mnist_config.json

taskset -c 2 maqao OV -R1 xp=final_reports/al_dense_report/pure_python -dbg=1 --replace --config=./configurations_files/python_al_dense_pure_python_config.json
taskset -c 2 maqao OV -R1 xp=final_reports/al_dense_report/cython      -dbg=1 --replace --config=./configurations_files/python_al_dense_cython_config.json
taskset -c 2 maqao OV -R1 xp=final_reports/al_dense_report/c_extension -dbg=1 --replace --config=./configurations_files/python_al_dense_c_extension_config.json

date