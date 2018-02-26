#!/bin/bash

pushd lib/smurff-cpp/cmake
rm -rf build 
mkdir build
cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DLAPACK_LIBRARIES=${CONDA_PREFIX}/lib/libmkl_rt${SHLIB_EXT} \
    -DOpenMP_CXX_FLAGS=-fopenmp=libiomp5 -DOpenMP_C_FLAGS=-fopenmp=libiomp5
make -j$CPU_COUNT
make install
popd

pushd python/smurff
$PYTHON setup.py install --with-openmp --single-version-externally-managed --record=record.txt
popd
