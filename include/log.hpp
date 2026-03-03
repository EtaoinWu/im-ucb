#pragma once

#include <iostream>
#include <string>
#include <string_view>

extern std::string identity;

inline auto my_log(std::string_view message) -> void {
  std::cout << identity << ": " << message << '\n';
}

inline auto set_identity(std::string_view id) -> void { identity = id; }
