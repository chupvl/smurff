rm -rf build
mkdir build
cd build
cmake ../ -DENABLE_OPENBLAS=ON -DCMAKE_BUILD_TYPE=Debug -DENABLE_VERBOSE_COMPILER_LOG=ON
make
Debug/tests
