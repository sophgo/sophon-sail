set(SAIL_INCLUDE_DIRS "/opt/sophon/sophon-sail/include" "/opt/sophon/sophon-sail/include/sail")
set(SAIL_LIB_DIRS "/opt/sophon/sophon-sail/lib")

find_library(SAIL_LIBRARY NAMES sail PATHS  NO_DEFAULT_PATH)

add_definitions(-DSAIL_WITH_BMRT_FLAG=1)
