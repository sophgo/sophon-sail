# set system name and processor type
set(CMAKE_SYSTEM_NAME        Linux)
set(CMAKE_SYSTEM_PROCESSOR   sw64)

# set cross compiler
set(CROSS_COMPILE sw_64-sunway-linux-gnu-)
# set(CROSS_COMPILE sw_64-linux-gnu-)

if (DEFINED TOOLCHAIN_BASIC_PATH)
    SET(ENV{PATH} ${TOOLCHAIN_BASIC_PATH}/bin:$ENV{PATH})
endif()

set(CMAKE_C_COMPILER         ${CROSS_COMPILE}gcc)
set(CMAKE_CXX_COMPILER       ${CROSS_COMPILE}g++)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -Wno-stringop-truncation -Wno-format-truncation -Wno-error=deprecated-declarations -ffunction-sections -fdata-sections -fno-strict-aliasing -ffp-contract=off -Wno-unused-function -fPIC --machine=sw6b" CACHE STRING "cflags")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wno-stringop-truncation -Wno-format-truncation -Wno-error=deprecated-declarations -fexceptions -funwind-tables -rdynamic -fno-short-enums -ffunction-sections -fdata-sections -fno-strict-aliasing -ffp-contract=off -std=c++11 -Wno-unused-function -fPIC --machine=sw6b" CACHE STRING "c++flags")

# search for programs in the build host dir
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# search lib and header in the target dir
set(CMAKE_FIND_ROOT_PATH_LIBRARY      ONLY)
set(CMAKE_FIND_ROOT_PATH_INCLUDE      ONLY)
