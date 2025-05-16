#pragma once

#include <iostream>
#include <string>

extern std::string identity;

inline void my_log(const std::string &message) {
  std::cout << identity << ": " << message << std::endl;
}

inline void set_identity(const std::string &id) { identity = id; }
