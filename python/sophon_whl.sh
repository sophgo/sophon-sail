#!/bin/bash

LOCAL_DIR=$(dirname $(readlink -f "$0"))
pushd $LOCAL_DIR

# wrap pcie new version
python3 setup.py bdist_wheel
if [[ $? != 0 ]];then
  echo "Failed to build sophon wheel"
  exit 1
fi
echo "---- setup sophon wheel"
rm -rf ./sophon*.egg-info ./build

# insert hooks to .git/hooks
SOURCE_DIR="../hooks"
TARGET_DIR="../.git/hooks"
if [ -d "$TARGET_DIR" ]; then
    echo "Install git hooks from $SOURCE_DIR to $TARGET_DIR"
    cp -pu "$SOURCE_DIR/"* "$TARGET_DIR/"
fi

popd