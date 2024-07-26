'''
Setup file for SOPHON
'''
from __future__ import print_function
import os
import shutil
from distutils.core import setup, Extension
from setuptools import find_packages

def is_only_runtime(file_name):
    ret = False
    with open(file_name) as fp:
        lines = fp.readlines()
        for line in lines:
            if len(line.split("ONLY_RUNTIME:"))>1:
                print(line)
                if len(line.split("=ON"))>1:
                    ret = True
                    break
                break
    return ret

# check sail pylib status
LIB_DIR = '../../build/lib/'
NATIVE_SO_PATH = '../../build/lib/sail.cpython-35m-x86_64-linux-gnu.so'
DST_PATH = './sophon'
CMAKECache_FILE="../../build/CMakeCache.txt"
for root,dirs,files in os.walk(LIB_DIR):
  for file in files:
    if file.split('.')[0] == 'sail':
      NATIVE_SO_PATH=os.path.join(root,file)

if os.path.exists(DST_PATH):
  shutil.rmtree(DST_PATH)

shutil.copytree("../../pyis","sophon")
pyi_name = "sophon/sail.pyi"
if is_only_runtime(CMAKECache_FILE):
  shutil.move("sophon/sail_onlyruntime.pyi",pyi_name)
  os.remove("sophon/_multimedia.pyi")
  os.remove("sophon/_high_performance.pyi")
else:
  os.remove("sophon/sail_onlyruntime.pyi")

shutil.copy("../__init__.py", "sophon/__init__.py")

if os.path.exists("./dist"):
  os.system("rm -f ./dist/*")

if os.path.exists(NATIVE_SO_PATH):
  print(NATIVE_SO_PATH.split('-')[2])
  try:
    shutil.copy(NATIVE_SO_PATH, DST_PATH)
  except shutil.SameFileError:
    pass
else:
  raise IOError("sail python lib not found")

# SOPHON python module
PACKAGES = ['sophon']

filehandle = open("../../git_version","r")
git_version = filehandle.readline().rstrip("\n").rstrip("\r")
print(git_version)

# wrap SOPHON python module
setup(name='sophon',
      version=git_version,
      description='Inference samples for deep learning on SOPHON products.',
      author='SOPHON algorithm team',
      url='https://github.com/sophon-ai-algo/sophon-inference',
      long_description='''
Guide to deploying deep-learning inference networks and deep vision primitives on SOPHON TPU.
''',
      packages=PACKAGES,
      include_package_data=True)
