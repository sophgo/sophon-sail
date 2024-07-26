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
LIB_DIR = '../../build/lib/Release/'
X86_PATH = '../../build/lib/Release/sail.cp38-win_amd64.pyd'
DST_PATH = './sophon'
CMAKECache_FILE="../../build/CMakeCache.txt"
for root,dirs,files in os.walk(LIB_DIR):
  for file in files:
    if file.split('.')[-1] == 'pyd':
      X86_PATH=os.path.join(root,file)

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

if os.path.exists(X86_PATH):
  print("x86")
  try:
    shutil.copy(X86_PATH, DST_PATH)
  except shutil.SameFileError:
    pass
else:
  raise IOError("sail python lib not found")

# sophon python module
PACKAGES = ['sophon']

filehandle = open("../../git_version","r")
git_version = filehandle.readline().rstrip("\n").rstrip("\r")
print(git_version)

# wrap sophon python module
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

