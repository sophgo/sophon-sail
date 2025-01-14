'''
Setup file for SOPHON
'''
from __future__ import print_function
import os
import shutil
from setuptools import setup, Extension, find_packages


def is_only_runtime(file_name):
    ret = False
    with open(file_name) as fp:
        lines = fp.readlines()
        for line in lines:
            if len(line.split("ONLY_RUNTIME:")) > 1:
                print(line)
                if len(line.split("=ON")) > 1:
                    ret = True
                    break
                break
    return ret


# check sail pylib status
RISCV64_PATH = '../../build/lib/sail.so'
DST_PATH = './sophon'
CMAKECache_FILE = "../../build/CMakeCache.txt"

filehandle = open("../../git_version", "r")
git_version = filehandle.readline().rstrip('\n').rstrip('\r')

if os.path.exists(DST_PATH):
    shutil.rmtree(DST_PATH)

shutil.copytree("../../pyis", "sophon")
pyi_name = "sophon/sail.pyi"
if is_only_runtime(CMAKECache_FILE):
    shutil.move("sophon/sail_onlyruntime.pyi", pyi_name)
    os.remove("sophon/_multimedia.pyi")
    os.remove("sophon/_high_performance.pyi")
else:
    os.remove("sophon/sail_onlyruntime.pyi")

shutil.copy("../__init__.py", "sophon/__init__.py")

if os.path.exists("./dist"):
    os.system("rm -f./dist/*")

if os.path.exists(RISCV64_PATH):
    try:
        shutil.copy(RISCV64_PATH, DST_PATH)
    except shutil.SameFileError:
        pass

    # sophon_aarch64 python module
    PACKAGES_RISCV64 = ['sophon']

    setup(name='sophon_riscv64',
          version=git_version,
          description='Inference samples for deep learning on SOPHON products.',
          author='SOPHON algorithm team',
          url='https://github.com/sophon-ai-algo/sophon-inference',
          long_description='''
  Guide to deploying deep-learning inference networks and deep vision primitives on SOPHON TPU.
  ''',
          packages=PACKAGES_RISCV64,
          include_package_data=True)
else:
    raise FileNotFoundError("sail lib not found")
