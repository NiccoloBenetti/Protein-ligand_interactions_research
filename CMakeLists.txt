cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(interaction VERSION 5.0 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

include_directories(${CUDAToolkit_INCLUDE_DIRS})

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 --expt-relaxed-constexpr --expt-extended-lambda")

# Imposta la compilazione separata CUDA
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "The type of build" FORCE)
  message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified")
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

# Esportare anche i comandi di compilazione
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Aggiungi flag per il compilatore
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options(-Wall -Wextra -Wpedantic -Wl,-z,defs -Wl,-z,now -Wl,-z,relro)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wdouble-promotion)
endif()

if ("${CMAKE_GENERATOR}" STREQUAL "Ninja")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        add_compile_options(-fdiagnostics-color=always)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        add_compile_options(-fcolor-diagnostics)
    endif()
endif()

# Trova RDKit
find_package(rdkit REQUIRED)

# Definisci i file sorgenti e header
set(header_path "${CMAKE_CURRENT_SOURCE_DIR}/src")
list(APPEND header_files
  "${header_path}/main.hpp"
)

set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/src")
list(APPEND source_files
  "${source_path}/main.cpp"
)

# Aggiungi l'eseguibile principale
add_executable(interaction ${header_files} ${source_files})

# Collega le librerie RDKit e CUDA all'eseguibile principale
target_link_libraries(interaction PUBLIC
  RDKitFileParsers
  RDKitGraphMol
  RDKitRDGeneral
  RDKitSmilesParse
  RDKitSubstructMatch
  RDKitFileParsers
  RDKitRDGeometryLib
  RDKitChemTransforms
  CUDA::cudart
)

# Imposta lo standard C++ e CUDA
set_target_properties(interaction PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  CUDA_STANDARD 11
  CUDA_STANDARD_REQUIRED ON
)

# Abilita le ottimizzazioni LTO (link-time)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
    set_property(TARGET interaction PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()


# Collega correttamente le librerie RDKit
get_target_property(RDKIT_LIB_FULLPATH RDKit::RDGeneral LOCATION)
get_filename_component(RDKIT_LIB_DIRPATH "${RDKIT_LIB_FULLPATH}" DIRECTORY)

cmake_path(GET RDKIT_LIB_DIRPATH PARENT_PATH RDKIT_INSTALL_PREFIX)

target_include_directories(interaction PUBLIC "${RDKIT_INSTALL_PREFIX}/Code")
target_link_directories(interaction PUBLIC "${RDKIT_INSTALL_PREFIX}/lib")
target_link_directories(interaction PUBLIC "${RDKIT_INSTALL_PREFIX}/lib64")



