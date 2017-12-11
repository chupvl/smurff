#pragma once

#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <memory>
#include <array>
#include <map>

#include <Eigen/Sparse>
#include <Eigen/Dense>



#include <SmurffCpp/Utils/Error.h>



#ifdef NDEBUG
#define SHOW(m)
#else
#define SHOW(m) std::cout << #m << ":\n" << m << std::endl;
#endif



