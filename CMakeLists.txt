cmake_minimum_required(VERSION 3.10.1) # for CUDA support
if(MSVC)
  add_compile_options(/W3 /WX)
else()
  add_compile_options(-W -Wall -Werror -I"/usr/local/cuda/include")
endif()
enable_language(CUDA)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fno-omit-frame-pointer")
find_package( BLAS )


add_library( lpsolver
  STATIC
  src/culiblp.cu
  src/cumatman.cu
  src/liblp.c
  src/matman.c)
target_include_directories( lpsolver PUBLIC include )
target_link_libraries( lpsolver PUBLIC  )

add_executable( cpu_solver src/lpsolver.c )
target_link_libraries( cpu_solver PRIVATE lpsolver )

add_executable( gpu_solver src/culpsolver.cpp )
target_link_libraries( gpu_solver PRIVATE lpsolver )

