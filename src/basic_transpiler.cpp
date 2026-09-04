#include "basic_transpiler.hpp"
#include <sstream>
#include <vector>
#include <regex>
#include <unordered_set>
#include <cctype>

namespace geochem {

static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (std::string::npos == first) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

static std::string to_lower(const std::string& str) {
    std::string out = str;
    for (char& c : out) c = std::tolower(c);
    return out;
}

std::string BasicTranspiler::transpile(const std::string& rate_name, const std::string& basic_code) {
    std::stringstream ss(basic_code);
    std::string line;
    std::vector<std::string> transpiled_lines;
    std::unordered_set<std::string> variables;

    // Pattern to catch assignments: extracts the variable name
    std::regex assign_regex(R"(^\s*(\d+)?\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=)");
    
    // Pattern to catch GOTO statements
    std::regex goto_regex(R"(goto\s+(\d+))", std::regex_constants::icase);
    std::regex if_then_regex(R"(if\s*\((.*?)\)\s*then\s*(.*))", std::regex_constants::icase);
    
    // Pattern to catch x ^ y
    std::regex pow_regex(R"(([\w\.\(\)\"\'_]+)\s*\^\s*([\w\.\(\)\"\'_\-]+))", std::regex_constants::icase);

    // Variable extraction pass
    while (std::getline(ss, line)) {
        std::string l = trim(line);
        // Strip # comments from end of line
        size_t hash_pos = l.find('#');
        if (hash_pos != std::string::npos) {
            l = l.substr(0, hash_pos);
        }
        l = trim(l);

        if (l.empty() || to_lower(l.substr(0, 3)) == "rem") {
            continue;
        }

        std::smatch match;
        if (std::regex_search(l, match, assign_regex)) {
            std::string var_name = match[2];
            // Don't declare variables that are built-in functions
            std::string lower_var = to_lower(var_name);
            if (lower_var != "m" && lower_var != "m0" && lower_var != "time" && lower_var != "rate") {
                variables.insert(var_name);
            }
        }
    }

    // Pass 2: Actual transpilation
    ss.clear();
    ss.str(basic_code);

    while (std::getline(ss, line)) {
        std::string l = trim(line);
        // Remove trailing # comments
        size_t hash_pos = l.find('#');
        if (hash_pos != std::string::npos) {
            l = l.substr(0, hash_pos);
        }
        l = trim(l);
        
        if (l.empty()) {
            continue;
        }
        
        // Remove REM comments
        std::string lower_l = to_lower(l);
        if (lower_l.find("rem ") == 0 || lower_l.find("rem\t") == 0 || lower_l == "rem") {
            continue;
        }

        std::string t_line = l;

        // 1. Line numbers -> Labels
        std::smatch label_match;
        if (std::regex_search(t_line, label_match, std::regex(R"(^(\d+)\s+(.*))"))) {
            t_line = "L" + label_match[1].str() + ": " + label_match[2].str();
        }

        // 2. AND / OR
        t_line = std::regex_replace(t_line, std::regex(R"(\band\b)", std::regex_constants::icase), "&&");
        t_line = std::regex_replace(t_line, std::regex(R"(\bor\b)", std::regex_constants::icase), "||");

        // 3. IF THEN
        std::smatch match;
        if (std::regex_search(t_line, match, if_then_regex)) {
            std::string prefix = match.prefix();
            std::string cond = match[1];
            std::string action = match[2];
            // Re-check for GOTO inside action
            action = std::regex_replace(action, goto_regex, "goto L$1;");
            if (!action.empty() && action.back() != ';') action += ";";
            t_line = prefix + "if (" + cond + ") { " + action + " }";
        } else {
            t_line = std::regex_replace(t_line, goto_regex, "goto L$1;");
        }

        // 4. Built-in functions & context variables
        t_line = std::regex_replace(t_line, std::regex(R"(\bsi\s*\()", std::regex_constants::icase), "ctx.si(");
        t_line = std::regex_replace(t_line, std::regex(R"(\bsr\s*\()", std::regex_constants::icase), "ctx.sr(");
        t_line = std::regex_replace(t_line, std::regex(R"(\bact\s*\()", std::regex_constants::icase), "ctx.act(");
        t_line = std::regex_replace(t_line, std::regex(R"(\bparm\s*\()", std::regex_constants::icase), "ctx.parm(");
        t_line = std::regex_replace(t_line, std::regex(R"(\blog10\s*\()", std::regex_constants::icase), "sycl::log10(");
        t_line = std::regex_replace(t_line, std::regex(R"(\bexp\s*\()", std::regex_constants::icase), "sycl::exp(");
        
        t_line = std::regex_replace(t_line, std::regex(R"(\bm\b)", std::regex_constants::icase), "ctx.m()");
        t_line = std::regex_replace(t_line, std::regex(R"(\bm0\b)", std::regex_constants::icase), "ctx.m0()");
        t_line = std::regex_replace(t_line, std::regex(R"(\btime\b)", std::regex_constants::icase), "ctx.time()");
        t_line = std::regex_replace(t_line, std::regex(R"(\btk\b)", std::regex_constants::icase), "ctx.tk()");

        // 5. Generic Exponents a^b -> sycl::pow(a, b)
        // Since C++ regex doesn't support recursive parenthesis matching, we do this manually.
        while (true) {
            size_t caret = t_line.find('^');
            if (caret == std::string::npos) break;

            // Find base 'a' (scan backwards)
            int start_a = caret - 1;
            int parens = 0;
            if (t_line[start_a] == ')') {
                parens = 1;
                start_a--;
                while (start_a >= 0 && parens > 0) {
                    if (t_line[start_a] == ')') parens++;
                    else if (t_line[start_a] == '(') parens--;
                    start_a--;
                }
            } else {
                while (start_a >= 0 && (std::isalnum(t_line[start_a]) || t_line[start_a] == '_' || t_line[start_a] == '.')) {
                    start_a--;
                }
            }
            start_a++; // first char of 'a'

            // Find exponent 'b' (scan forwards)
            int end_b = caret + 1;
            parens = 0;
            if (end_b < t_line.length() && t_line[end_b] == '(') {
                parens = 1;
                end_b++;
                while (end_b < t_line.length() && parens > 0) {
                    if (t_line[end_b] == '(') parens++;
                    else if (t_line[end_b] == ')') parens--;
                    end_b++;
                }
            } else {
                if (end_b < t_line.length() && (t_line[end_b] == '-' || t_line[end_b] == '+')) {
                    end_b++;
                }
                while (end_b < t_line.length() && (std::isalnum(t_line[end_b]) || t_line[end_b] == '_' || t_line[end_b] == '.')) {
                    end_b++;
                }
            }
            
            std::string a = t_line.substr(start_a, caret - start_a);
            std::string b = t_line.substr(caret + 1, end_b - caret - 1);
            
            std::string replacement;
            if (to_lower(a) == "e") {
                replacement = "sycl::exp(static_cast<Real>(" + b + "))";
            } else {
                replacement = "sycl::pow(static_cast<Real>(" + a + "), static_cast<Real>(" + b + "))";
            }
            
            t_line = t_line.substr(0, start_a) + replacement + t_line.substr(end_b);
        }

        // 6. SAVE
        if (std::regex_search(t_line, std::regex(R"(\bsave\s+)", std::regex_constants::icase))) {
            t_line = std::regex_replace(t_line, std::regex(R"(\bsave\s+(.*))", std::regex_constants::icase), "return $1;");
        } else if (!t_line.empty() && t_line.back() != ';' && t_line.back() != '}') {
            // Add semicolon to statements
            t_line += ";";
        }

        transpiled_lines.push_back("    " + t_line);
    }

    std::string out = "#include <sycl/sycl.hpp>\n#include \"rates.hpp\"\n\n";
    out += "namespace geochem {\n";
    out += "template<typename Real>\nSYCL_EXTERNAL inline Real rate_" + rate_name + "(const geochem::RateContext<Real>& ctx) {\n";
    
    // Declare all variables
    if (!variables.empty() || true) {
        out += "    Real ";
        bool first = true;
        for (const auto& v : variables) {
            if (!first) out += ", ";
            out += v + " = 0.0";
            first = false;
        }
        if (!first) out += ", ";
        out += "rate = 0.0;\n"; // always declare rate just in case
    }

    for (const auto& tl : transpiled_lines) {
        out += tl + "\n";
    }

    out += "    return 0.0;\n}\n} // namespace geochem\n";
    return out;
}

} // namespace geochem
