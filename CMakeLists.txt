cmake_minimum_required(VERSION 3.10.1) # for CUDA support
project(LpSolver LANGUAGES C CXX CUDA)
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /WX")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Werror")
endif()
# enable_language(CUDA)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fno-omit-frame-pointer")
#set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Xcompiler -fno-omit-frame-pointer")
#set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -Xcompiler -fno-omit-frame-pointer")
find_package( BLAS REQUIRED )
find_package( gflags REQUIRED )
find_package( glog REQUIRED )
find_package( GTest REQUIRED )

add_library( lpsolver
  STATIC
  src/culiblp.cu
  src/cumatman.cu
  src/liblp.c
  src/matman.c)

target_include_directories( lpsolver PUBLIC include ${BLAS_INCLUDE_DIR} ${GFALGS_INCLUDE_DIR} )
target_link_libraries( lpsolver PUBLIC -lcublas ${BLAS_LIBRARIES} ${GFLAGS_LIBRARIES} glog::glog ${GTEST_LIBRARIES})
target_compile_features( lpsolver PUBLIC cxx_std_11 )

add_executable( cpu_solver src/lpsolver.c )
target_link_libraries( cpu_solver PRIVATE lpsolver )

add_executable( gpu_solver src/culpsolver.cu )
target_link_libraries( gpu_solver PRIVATE lpsolver )

add_executable( jlpsolver src/jlpsolver.cc )
target_link_libraries( jlpsolver PRIVATE lpsolver )

# add_executable( cuda_jlpsolver src/cuda_jlpsolver.cu )
# target_link_libraries( cuda_jlpsolver PRIVATE lpsolver )

add_executable( popmat src/popmat.c )
