'''
Setup file for SOPHON
'''

import os
import sys
import shutil
# from distutils.core import setup, Extension
from setuptools import find_packages
from setuptools import setup

def merge_pyis(file_list,save_name="sail.pyi"):
    with open(save_name,"w+",encoding='utf-8') as fp:
        lines_head = []
        lines_body = []
        for file_name in file_list:
            with open(file_name,"+r",encoding='utf-8') as fp_sub:
                lines_temp = fp_sub.readlines()
                for line in lines_temp:
                    if line.find("import") == 0 or line.find("from") == 0:
                        if line not in lines_head:
                            if line.find(".") < 0: 
                                lines_head.append(line)
                    else:
                        lines_body.append(line)
                lines_body.append("\n")
                lines_body.append("\n")
                fp_sub.close()
        lines_head = lines_head+lines_body
        fp.writelines(lines_head);
        fp.close()

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

def is_only_runtime(file_name):
    ret = False
    with open(file_name) as fp:
        lines = fp.readlines()
        for line in lines:
            if len(line.split("ONLY_RUNTIME:"))>1:
                if len(line.split("=ON"))>1:
                    ret = True
                    break
                break
    fp.close()
    return ret

def get_build_type(CMakeCachefile):
    ret = False
    with open(CMakeCachefile) as fp:
        lines = fp.readlines()
        #使用交叉编译
        for line in lines:
            if len(line.split("BUILD_TYPE:UNINITIALIZED"))>1:  
                key_temp=line.rstrip().split("=")[-1]
                if key_temp == "soc":
                    fp.close()
                    return "SOC"
                if key_temp == "arm_pcie":
                    fp.close()
                    return "ARM_PCIE"
                if key_temp == "loongarch":
                    fp.close()
                    return "LOONGARCH"
                if key_temp == "riscv":
                    fp.close()
                    return "RISICV"
                if key_temp == "sw64" or key_temp == "sw_64":
                    fp.close()
                    return "SW_64"
                
        #未使用交叉编译的情况
        for line in lines:
            if len(line.split("PYTHON_MODULE_EXTENSION:INTERNAL="))>1:         
                key_temp=line.split("=")[-1]
                if key_temp.find("x86_64") > 0:
                    fp.close()
                    return "PCIE"
                elif key_temp.find("sw_64") > 0:
                    fp.close()
                    return "SW_64"
                elif key_temp.find("aarch64") > 0:
                    fp.close()
                    mode = "ARM_PCIE"
                    with open("/proc/cpuinfo","r") as cpuinfo_fp:
                        line_cpu_info = cpuinfo_fp.readlines()
                        for line_temp in line_cpu_info:
                            if line_temp.find("model name") >= 0:
                                if line_temp.find("bm168") >= 0 or line_temp.find("cv186") >= 0:
                                    mode = "SOC"
                        cpuinfo_fp.close()
                    fp.close()
                    return mode
                elif key_temp.find("loongarch64") > 0:
                    return "LOONGARCH"
                elif key_temp.find("riscv64") > 0:
                    return "RISICV"
    
    return "unkonwn"



if __name__ == "__main__":

    current_folder = os.path.dirname(os.path.abspath(__file__))
    build_path = os.path.join(current_folder,"../build")

    CMAKECache_FILE=os.path.join(build_path,"CMakeCache.txt") 
    if not os.path.exists(CMAKECache_FILE):
        raise FileNotFoundError("Can not find cmake ceche file:",CMAKECache_FILE)
    
    build_result_path=os.path.join(build_path,"lib")
    build_so_path="sail.so"
    for root,dirs,files in os.walk(build_result_path):
        for file in files:
            if file.split('.')[0] == 'sail':
                build_so_path=os.path.join(root,file)

    dst_path=os.path.join(current_folder,"sophon")
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    os.makedirs(dst_path,exist_ok=True)

    shutil.copy(os.path.join(current_folder,"__init__.py"), dst_path)
    if os.path.exists(build_so_path):
        try:
            shutil.copy(build_so_path, dst_path)
        except shutil.SameFileError:
            pass
    else:
        raise IOError("sail python lib not found")


    pyi_list = []
    pyi_list.append(os.path.join(current_folder,"../pyis/_basic.pyi"))
    pyi_list.append(os.path.join(current_folder,"../pyis/_engine.pyi"))
    pyi_list.append(os.path.join(current_folder,"../pyis/_kernel_api.pyi"))
    pyi_list.append(os.path.join(current_folder,"../pyis/_algokit.pyi"))
    pyi_list.append(os.path.join(current_folder,"../pyis/_engine_llm.pyi"))
    if not is_only_runtime(CMAKECache_FILE):
        pyi_list.append(os.path.join(current_folder,"../pyis/_multimedia.pyi"))
        pyi_list.append(os.path.join(current_folder,"../pyis/_high_performance.pyi"))

    pyi_save_name=os.path.join(dst_path,"sail.pyi")
    merge_pyis(pyi_list,pyi_save_name)

    filehandle = open(os.path.join(current_folder,"../git_version"),"r")
    git_version = filehandle.readline().rstrip("\n").rstrip("\r")


    # SOPHON python module
    PACKAGES = ['sophon']
    module_name = "sophon"
    build_type=get_build_type(CMAKECache_FILE)
    if build_type == "SOC":
        module_name = "sophon_arm"
    elif build_type == "ARM_PCIE":
        module_name = "sophon_arm_pcie"
    elif build_type == "LOONGARCH":
        module_name = "sophon_loongarch64"
    elif build_type == "SW_64":
        module_name = "sophon_sw64"
    elif build_type == "RISICV":
        module_name = "sophon_riscv64"

    # wrap SOPHON python module
    setup(name=module_name,
        version=git_version,
        description='Inference samples for deep learning on SOPHON products.',
        author='SOPHON algorithm team',
        url='https://github.com/sophgo/sophon-sail',
        long_description='''
    Guide to deploying deep-learning inference networks and deep vision primitives on SOPHON TPU.
    ''',
        packages=PACKAGES,
        include_package_data=True,
        install_requires = ['numpy>=1']
    )


# current_folder = os.path.dirname(os.path.abspath(__file__))
# print(current_folder)


# merge_only_runtime()
# merge_all()
        

