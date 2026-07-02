set(SAIL_INCLUDE_DIRS "/workspace/sdk/sophon-sail/build_soc/sophon-sail/include" "/workspace/sdk/sophon-sail/build_soc/sophon-sail/include/sail")
set(SAIL_LIB_DIRS "/workspace/sdk/sophon-sail/build_soc/sophon-sail/lib")

find_library(SAIL_LIBRARY NAMES sail PATHS  NO_DEFAULT_PATH)

add_definitions(-DSAIL_WITH_BMRT_FLAG=1)
