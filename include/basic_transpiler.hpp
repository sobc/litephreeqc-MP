#pragma once

#include <string>

namespace geochem {

class BasicTranspiler {
public:
    static std::string transpile(const std::string& rate_name, const std::string& basic_code);
};

} // namespace geochem
