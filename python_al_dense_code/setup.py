from setuptools import setup, Extension, Command
from Cython.Build import cythonize
import shutil
import os

class CleanCommand(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vr ./build')
        shutil.rmtree('build', ignore_errors=True)


module = Extension(
            name               = 'al_dense_c_extension_lib', 
            sources            = ['al_dense_c_extension_lib.c'], 
            extra_compile_args = ['-O3', '-march=native', '-g', '-fno-omit-frame-pointer', '-funroll-loops']
        )


setup(
    description = 'Basic dense linear algebra librairie in c extension',
    ext_modules = [module],
    cmdclass    = { 'clean': CleanCommand },
)

ext_modules = [
    Extension(
        name               = "al_dense_cython_lib",
        sources            = ["al_dense_cython_lib.pyx"],
        extra_compile_args = ['-O3', '-march=native', '-g', '-fno-omit-frame-pointer', '-funroll-loops'], # => mean we use python3,
        ## Not need anymore
        # include_dirs       = [numpy.get_include()],
        # define_macros      = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    )
]

setup(
    name="cython_al_dense",
    ext_modules=cythonize(ext_modules, language_level="3")
)