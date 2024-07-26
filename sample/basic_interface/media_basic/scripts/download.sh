#!/bin/bash
res=$(which 7z)
if [ $? != 0 ]; then
    echo "Please install 7z on your system!"
    exit
fi

pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
scripts_dir=$(dirname $(readlink -f "$0"))

pushd $scripts_dir


if [ ! -d "../datasets" ]; 
then
    mkdir -p ../datasets

    python3 -m dfss --url=open@sophgo.com:sophon-sail/sample/test_pics.7z
    7z x test_pics.7z -o../datasets
    rm test_pics.7z
    echo "pics download!"
    
    python3 -m dfss --url=open@sophgo.com:sophon-sail/sample/test_car_person_1080P.7z
    7z x test_car_person_1080P.7z -o../datasets
    rm test_car_person_1080P.7z
    echo "videos download!"
else
    echo "Datasets folder exist! Remove it if you need to update."
fi
