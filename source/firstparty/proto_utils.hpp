// // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DmitriBogdanov/prototyping_utils ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// //
// // Macro-Module:  UTL_LOG
// // Documentation: https://github.com/DmitriBogdanov/prototyping_utils/blob/master/docs/MACRO_LOG.md
// // Source repo:   https://github.com/DmitriBogdanov/prototyping_utils
// //
// // This project is licensed under the MIT License
// //
// // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// #if !defined(UTL_PICK_MODULES) || defined(UTLMACRO_LOG)
// #ifndef UTLHEADERGUARD_LOG
// #define UTLHEADERGUARD_LOG

// // _______________________ INCLUDES _______________________

// #include <iostream>    // cout
// #include <ostream>     // ostream
// #include <string_view> // string_view

// // ____________________ DEVELOPER DOCS ____________________

// // A very minimalistic logger with source file info. Mainly used for debugging.
// // NOTE: Perhaps can be expanded to something faster and more production-usable.
// //
// // # UTL_LOG_SET_OUTPUT() #
// // Select ostream used by LOG macros.
// //
// // # UTL_LOG(), UTL_LOG_DEBUG() #
// // Print message to selected ostream prefixed with [<filename>:<line> (<function>)].
// // Accepts multiple args (with defined operator <<) that get concatenated into a single message.
// // DEBUG version compiles to nothing in release.

// // ____________________ IMPLEMENTATION ____________________

// // ======================
// // --- Logging Macros ---
// // ======================

// inline std::ostream* _utl_log_ostream = &std::cout;

// #define UTL_LOG_SET_OUTPUT(new_stream_) _utl_log_ostream = &new_stream_;

// template <typename... Args>
// inline void _utl_log_print(std::string_view file, int line, std::string_view func, const Args&... args) {
//     const std::string_view filename = file.substr(file.find_last_of("/\\") + 1);

//     ///(*_utl_log_ostream) << "\033[31;1m"; // Supported by Linux and Windows10+, but prints to files, figure out a fix
//     (*_utl_log_ostream) << "[" << filename << ":" << line << ", " << func << "()]";
//     ///(*_utl_log_ostream) << "\033[0m";

//     (*_utl_log_ostream) << " ";
//     ((*_utl_log_ostream) << ... << args);
//     (*_utl_log_ostream) << '\n';
// }

// #define UTL_LOG(...) _utl_log_print(__FILE__, __LINE__, __func__, __VA_ARGS__)

// #ifdef _DEBUG
// #define UTL_LOG_DEBUG(...) _utl_log_print(__FILE__, __LINE__, __func__, __VA_ARGS__)
// #else
// #define UTL_LOG_DEBUG(...)
// #endif

// #endif
// #endif // macro-module UTL_LOG






// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DmitriBogdanov/prototyping_utils ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Module:        utl::json
// Documentation: https://github.com/DmitriBogdanov/prototyping_utils/blob/master/docs/module_json.md
// Source repo:   https://github.com/DmitriBogdanov/prototyping_utils
//
// This project is licensed under the MIT License
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#if !defined(UTL_PICK_MODULES) || defined(UTLMODULE_JSON)
#ifndef UTLHEADERGUARD_JSON
#define UTLHEADERGUARD_JSON

// _______________________ INCLUDES _______________________

#include <array>            // array<>
#include <charconv>         // to_chars(), from_chars()
#include <cmath>            // isfinite()
#include <codecvt>          // codecvt_utf8<>
#include <cuchar>           // size_t, char32_t, mbstate_t
#include <fstream>          // ifstream, ofstream
#include <initializer_list> // initializer_list<>
#include <limits>           // numeric_limits<>::max_digits10, numeric_limits<>::max_exponent10
#include <map>              // map<>
#include <stdexcept>        // runtime_error
#include <string>           // string, stoul()
#include <string_view>      // string_view
#include <system_error>     // errc()
#include <type_traits>      // enable_if_t<>, void_t, is_convertible_v<>, is_same_v<>
#include <utility>          // move(), declval<>()
#include <variant>          // variant<>
#include <vector>           // vector<>

// ____________________ DEVELOPER DOCS ____________________

// TODO:

// ____________________ IMPLEMENTATION ____________________

namespace utl::json {

// ===================
// --- Misc. utils ---
// ===================

// Codepoint -> UTF-8 string conversion using <codecvt> (deprecated in C++17, removed in C++26)
// This is kinda horrible, but there doesn't seem to a better way.
// Returns success so we can handle the error message inside the parser itself.
inline bool _unicode_codepoint_to_utf8(std::string& destination, char32_t cp) {
    std::array<char, 4> buffer; // here we will put UTF-8 string
    char32_t const*     from = &cp;
    char*               end_of_buffer;

    std::mbstate_t              state;
    std::codecvt_utf8<char32_t> codecvt;

    if (codecvt.out(state, from, from + 1, from, buffer.data(), buffer.data() + 4, end_of_buffer)) return false;

    destination.append(buffer.data(), end_of_buffer);
    return true;
}

inline std::string _read_file_to_string(const std::string& path) {
    using namespace std::string_literals;

    // This seems the to be the fastest way of reading a text file
    // into 'std::string' without invoking OS-specific methods
    // See this StackOverflow thread:
    // [https://stackoverflow.com/questions/32169936/optimal-way-of-reading-a-complete-file-to-a-string-using-fstream]
    // And attached benchmarks:
    // [https://github.com/Sqeaky/CppFileToStringExperiments]

    std::ifstream file(path, std::ios::ate); // open file and immediately seek to the end
    if (!file.good()) throw std::runtime_error("Could not open file {"s + path + "."s);
    // NOTE: Add a way to return errors if file doesn't exist, exceptions aren't particularly good here

    const auto file_size = file.tellg(); // returns cursor pos, which is the end of file
    file.seekg(std::ios::beg);           // seek to the beginning
    std::string chars(file_size, 0);     // allocate string of appropriate size
    file.read(chars.data(), file_size);  // read into the string
    return chars;
}

template <typename T>
constexpr int _log_10_ceil(T num) {
    return num < 10 ? 1 : 1 + _log_10_ceil(num / 10);
}

std::string _pretty_error(std::size_t cursor, const std::string& chars) {
    // Special case for empty buffers
    if (chars.empty()) return "";

    // "Normalize" cursor if it's at the end of the buffer
    if (cursor >= chars.size()) cursor = chars.size() - 1;

    // Get JSON line number
    std::size_t line_number = 1; // don't want to include <algorithm> just for a single std::count()
    for (std::size_t pos = 0; pos < cursor; ++pos)
        if (chars[pos] == '\n') ++line_number;

    // Get contents of the current line
    constexpr std::size_t max_left_width  = 24;
    constexpr std::size_t max_right_width = 24;

    std::size_t line_start = cursor;
    for (; line_start > 0; --line_start)
        if (chars[line_start - 1] == '\n' || cursor - line_start >= max_left_width) break;

    std::size_t line_end = cursor;
    for (; line_end < chars.size() - 1; ++line_end)
        if (chars[line_end + 1] == '\n' || line_end - cursor >= max_right_width) break;

    const std::string_view line_contents(chars.data() + line_start, line_end - line_start + 1);

    // Format output
    const std::string line_prefix =
        "Line " + std::to_string(line_number) + ": "; // fits into SSO buffer in almost all cases

    std::string res;
    res.reserve(7 + 2 * line_prefix.size() + 2 * line_contents.size());

    res += '\n';
    res += line_prefix;
    res += line_contents;
    res += '\n';
    res.append(line_prefix.size(), ' ');
    res.append(cursor - line_start, '-');
    res += '^';
    res.append(line_end - cursor, '-');
    res += " [!]";

    return res;
}

// Workaround for 'static_assert(false)' making program ill-formed even
// when placed inide an 'if constexpr' branch that never compiles.
// 'static_assert(_always_false_v<T)' on the the other hand doesn't,
// which means we can use it to mark branches that should never compile.
template <class>
inline constexpr bool _always_false_v = false;

// Type trait that checks existence of '.begin()', 'end()' members
template <typename Type, typename = void, typename = void>
struct _is_const_iterable_through : std::false_type {};

template <typename Type>
struct _is_const_iterable_through<Type, std::void_t<decltype(++std::declval<Type>().begin())>,
                                  std::void_t<decltype(std::declval<Type>().end())>> : std::true_type {};

// Type trait that checks existence of '::key_type', '::mapped_type' members
template <typename Type, typename = void, typename = void>
struct _is_assotiative : std::false_type {};

template <typename Type>
struct _is_assotiative<Type, std::void_t<typename Type::key_type>, std::void_t<typename Type::mapped_type>>
    : std::true_type {};

// ===================================
// --- JSON type conversion traits ---
// ===================================

template <class T>
using _object_type_impl = std::map<std::string, T, std::less<>>;
// 'std::less<>' declares map as transparent, which means we can `.find()` for `std::string_view` keys
template <class T>
using _array_type_impl  = std::vector<T>;
using _string_type_impl = std::string;
using _number_type_impl = double;
using _bool_type_impl   = bool;
struct _null_type_impl {
    [[nodiscard]] bool operator==(const _null_type_impl&) const noexcept {
        return true;
    } // so we can check 'Null == Null'
};

template <class T>
constexpr bool is_object_like_v = _is_const_iterable_through<T>::value && _is_assotiative<T>::value;
// NOTE: Also check for 'key_type' being convertible to 'std::string'
template <class T>
constexpr bool is_array_like_v = _is_const_iterable_through<T>::value;
template <class T>
constexpr bool is_string_like_v = std::is_convertible_v<T, std::string_view>;
template <class T>
constexpr bool is_numeric_like_v = std::is_convertible_v<T, _number_type_impl>;
template <class T>
constexpr bool is_bool_like_v = std::is_same_v<T, _bool_type_impl>;
template <class T>
constexpr bool is_null_like_v = std::is_same_v<T, _null_type_impl>;

template <class T>
constexpr bool is_json_type_convertible_v = is_object_like_v<T> || is_array_like_v<T> || is_string_like_v<T> ||
                                            is_numeric_like_v<T> || is_bool_like_v<T> || is_null_like_v<T>;

// ==================
// --- Node class ---
// ==================

enum class Format { PRETTY, MINIMIZED };

class Node;
inline void _serialize_json_to_buffer(std::string& chars, const Node& node, Format format);

class Node {
public:
    using object_type = _object_type_impl<Node>;
    using array_type  = _array_type_impl<Node>;
    using string_type = _string_type_impl;
    using number_type = _number_type_impl;
    using bool_type   = _bool_type_impl;
    using null_type   = _null_type_impl;

private:
    using variant_type = std::variant<null_type, object_type, array_type, string_type, number_type, bool_type>;
    // 'null_type' should go first to ensure default-initialization creates 'null' nodes

    variant_type data{};

public:
    // -- Getters --
    // -------------

    template <class T>
    [[nodiscard]] T& get() {
        return std::get<T>(this->data);
    }

    template <class T>
    [[nodiscard]] const T& get() const {
        return std::get<T>(this->data);
    }

    [[nodiscard]] object_type& get_object() { return this->get<object_type>(); }
    [[nodiscard]] array_type&  get_array() { return this->get<array_type>(); }
    [[nodiscard]] string_type& get_string() { return this->get<string_type>(); }
    [[nodiscard]] number_type& get_number() { return this->get<number_type>(); }
    [[nodiscard]] bool_type&   get_bool() { return this->get<bool_type>(); }
    [[nodiscard]] null_type&   get_null() { return this->get<null_type>(); }

    [[nodiscard]] const object_type& get_object() const { return this->get<object_type>(); }
    [[nodiscard]] const array_type&  get_array() const { return this->get<array_type>(); }
    [[nodiscard]] const string_type& get_string() const { return this->get<string_type>(); }
    [[nodiscard]] const number_type& get_number() const { return this->get<number_type>(); }
    [[nodiscard]] const bool_type&   get_bool() const { return this->get<bool_type>(); }
    [[nodiscard]] const null_type&   get_null() const { return this->get<null_type>(); }

    template <class T>
    [[nodiscard]] bool is() const noexcept {
        return std::holds_alternative<T>(this->data);
    }

    [[nodiscard]] bool is_object() const noexcept { return this->is<object_type>(); }
    [[nodiscard]] bool is_array() const noexcept { return this->is<array_type>(); }
    [[nodiscard]] bool is_string() const noexcept { return this->is<string_type>(); }
    [[nodiscard]] bool is_number() const noexcept { return this->is<number_type>(); }
    [[nodiscard]] bool is_bool() const noexcept { return this->is<bool_type>(); }
    [[nodiscard]] bool is_null() const noexcept { return this->is<null_type>(); }

    template <class T>
    [[nodiscard]] T* get_if() noexcept {
        return std::get_if<T>(&this->data);
    }

    template <class T>
    [[nodiscard]] const T* get_if() const noexcept {
        return std::get_if<T>(&this->data);
    }

    // -- Object methods --
    // --------------------

    Node& operator[](std::string_view key) {
        // 'std::map<K, V>::operator[]()' and 'std::map<K, V>::at()' don't support
        // support heterogeneous lookup, we have to reimplement them manually
        if (this->is_null()) this->data = object_type(); // only 'null' converts to object automatically on 'json[key]'
        auto& object = this->get_object();
        auto  it     = object.find(key);
        if (it == object.end()) it = object.emplace(key, Node{}).first;
        return it->second;
    }

    [[nodiscard]] const Node& operator[](std::string_view key) const {
        // 'std::map<K, V>::operator[]()' and 'std::map<K, V>::at()' don't support
        // support heterogeneous lookup, we have to reimplement them manually
        const auto& object = this->get_object();
        const auto  it     = object.find(key);
        if (it == object.end())
            throw std::runtime_error("Accessing non-existent key {" + std::string(key) + "} in JSON object.");
        return it->second;
    }

    [[nodiscard]] Node& at(std::string_view key) {
        // Non-const 'operator[]' inserts non-existent keys, '.at()' should throw instead
        auto&      object = this->get_object();
        const auto it     = object.find(key);
        if (it == object.end())
            throw std::runtime_error("Accessing non-existent key {" + std::string(key) + "} in JSON object.");
        return it->second;
    }

    [[nodiscard]] const Node& at(std::string_view key) const { return this->operator[](key); }

    [[nodiscard]] bool contains(std::string_view key) const {
        const auto& object = this->get_object();
        const auto  it     = object.find(std::string(key));
        return it != object.end();
    }

    template <class T>
    [[nodiscard]] const T& value_or(std::string_view key, const T& else_value) {
        const auto& object = this->get_object();
        const auto  it     = object.find(std::string(key));
        if (it != object.end()) return it->second.get<T>();
        return else_value;
        // same thing as 'this->contains(key) ? json.at(key).get<T>() : else_value' but without a second map lookup
    }

    // -- Assignment --
    // ----------------

    // Converting assignment
    template <class T, std::enable_if_t<
                           !std::is_same_v<std::decay_t<T>, Node> && !std::is_same_v<std::decay_t<T>, object_type> &&
                               !std::is_same_v<std::decay_t<T>, array_type> &&
                               !std::is_same_v<std::decay_t<T>, string_type> && is_json_type_convertible_v<T>,
                           bool> = true>
    Node& operator=(const T& value) {
        // Don't take types that decay to Node/object/array/string to prevent
        // shadowing native copy/move assignment for those types

        constexpr bool is_object_like  = _is_const_iterable_through<T>::value && _is_assotiative<T>::value;
        constexpr bool is_array_like   = _is_const_iterable_through<T>::value;
        constexpr bool is_string_like  = std::is_convertible_v<T, std::string_view>;
        constexpr bool is_numeric_like = std::is_convertible_v<T, number_type>;
        constexpr bool is_bool_like    = std::is_same_v<T, bool_type>;
        constexpr bool is_null_like    = std::is_same_v<T, null_type>;

        // Several "type-like' characteristics can be true at the same time,
        // to resolve ambiguity we assign the following conversion priority:
        // string > object > array > bool > null > numeric

        if constexpr (is_string_like) {
            this->data.emplace<string_type>(value);
        } else if constexpr (is_object_like) {
            this->data.emplace<object_type>();
            auto& object = this->get_object();
            for (const auto& [key, val] : value) object[key] = val;
        } else if constexpr (is_array_like) {
            this->data.emplace<array_type>();
            auto& array = this->get_array();
            for (const auto& elem : value) array.emplace_back(elem);
        } else if constexpr (is_bool_like) {
            this->data.emplace<bool_type>(value);
        } else if constexpr (is_null_like) {
            this->data.emplace<null_type>(value);
        } else if constexpr (is_numeric_like) {
            this->data.emplace<number_type>(value);
        } else {
            static_assert(_always_false_v<T>, "Method is a non-exhaustive visitor of std::variant<>.");
        }

        return *this;
    }

    // "native" copy/move semantics for types that support it
    Node& operator=(const object_type& value) {
        this->data = value;
        return *this;
    }
    Node& operator=(object_type&& value) {
        this->data = std::move(value);
        return *this;
    }

    Node& operator=(const array_type& value) {
        this->data = value;
        return *this;
    }
    Node& operator=(array_type&& value) {
        this->data = std::move(value);
        return *this;
    }

    Node& operator=(const string_type& value) {
        this->data = value;
        return *this;
    }
    Node& operator=(string_type&& value) {
        this->data = std::move(value);
        return *this;
    }

    // Support for 'std::initializer_list' type deduction,
    // (otherwise the call is ambiguous)
    template <class T>
    Node& operator=(std::initializer_list<T> ilist) {
        // We can't just do 'return *this = array_type(value);' because compiler doesn't realize it can
        // convert 'std::initializer_list<T>' to 'std::vector<Node>' for all 'T' convertable to 'Node',
        // we have to invoke 'Node()' constructor explicitly (here it happens in 'emplace_back()')
        array_type array_value;
        array_value.reserve(ilist.size());
        for (const auto& e : ilist) array_value.emplace_back(e);
        this->data = std::move(array_value);
        return *this;
    }

    template <class T>
    Node& operator=(std::initializer_list<std::initializer_list<T>> ilist) {
        // Support for 2D brace initialization
        array_type array_value;
        array_value.reserve(ilist.size());
        for (const auto& e : ilist) {
            array_value.emplace_back();
            array_value.back() = e;
        }
        // uses 1D 'operator=(std::initializer_list<T>)' to fill each node of the array
        this->data = std::move(array_value);
        return *this;
    }

    template <class T>
    Node& operator=(std::initializer_list<std::initializer_list<std::initializer_list<T>>> ilist) {
        // Support for 3D brace initialization
        // it's dumb, but it works
        array_type array_value;
        array_value.reserve(ilist.size());
        for (const auto& e : ilist) {
            array_value.emplace_back();
            array_value.back() = e;
        }
        // uses 2D 'operator=(std::initializer_list<std::initializer_list<T>>)' to fill each node of the array
        this->data = std::move(array_value);
        return *this;
    }

    // we assume no reasonable person would want to type a 4D+ array as 'std::initializer_list<>',
    // if they really want to they can specify the type of the top layer and still be fine

    // -- Constructors --
    // ------------------

    Node& operator=(const Node&) = default;
    Node& operator=(Node&&)      = default;

    Node()            = default;
    Node(const Node&) = default;
    Node(Node&&)      = default;

    // Converting ctor
    template <class T, std::enable_if_t<
                           !std::is_same_v<std::decay_t<T>, Node> && !std::is_same_v<std::decay_t<T>, object_type> &&
                               !std::is_same_v<std::decay_t<T>, array_type> &&
                               !std::is_same_v<std::decay_t<T>, string_type> && is_json_type_convertible_v<T>,
                           bool> = true>
    Node(const T& value) {
        *this = value;
    }

    Node(const object_type& value) { this->data = value; }
    Node(object_type&& value) { this->data = std::move(value); }
    Node(const array_type& value) { this->data = value; }
    Node(array_type&& value) { this->data = std::move(value); }
    Node(std::string_view value) { this->data = string_type(value); }
    Node(const string_type& value) { this->data = value; }
    Node(string_type&& value) { this->data = std::move(value); }
    Node(number_type value) { this->data = value; }
    Node(bool_type value) { this->data = value; }
    Node(null_type value) { this->data = value; }

    // --- JSON Serializing public API ---
    // -----------------------------------

    [[nodiscard]] std::string to_string(Format format = Format::PRETTY) const {
        std::string buffer;
        _serialize_json_to_buffer(buffer, *this, format);
        return buffer;
    }

    void to_file(const std::string& filepath, Format format = Format::PRETTY) {
        auto chars = this->to_string(format);
        std::ofstream(filepath).write(chars.data(), chars.size());
        // maybe a little faster than doing 'std::ofstream(filepath) << node.to_string(format)'
    }
};

// Public typedefs
using Object = Node::object_type;
using Array  = Node::array_type;
using String = Node::string_type;
using Number = Node::number_type;
using Bool   = Node::bool_type;
using Null   = Node::null_type;

// =====================
// --- Lookup Tables ---
// =====================

constexpr std::size_t _number_of_char_values = 256;
// always true since 'sizeof(char) == 1' is guaranteed by the standard

// Lookup table used to check if number should be escaped and get a replacement char on at the same time.
// This allows us to replace multiple checks and if's with a single array lookup that.
//
// Instead of:
//    if (c == '"') { chars += '"' }
//    ...
//    else if (c == '\t') { chars += 't' }
// we get:
//    if (const char replacement = _lookup_serialized_escaped_chars[c]) { chars += replacement; }
//
// which ends up being a bit faster.
//
constexpr std::array<char, _number_of_char_values> _lookup_serialized_escaped_chars = [] {
    std::array<char, _number_of_char_values> res{};
    // default-initialized chars get initialized to '\0',
    // as far as I understand ('\0' == 0) is mandated by the standard,
    // which is why we can use it inside an 'if' condition
    res['"']  = '"';
    res['\\'] = '\\';
    // res['/']  = '/'; escaping forward slash in JSON is allowed, but redundant
    res['\b'] = 'b';
    res['\f'] = 'f';
    res['\n'] = 'n';
    res['\r'] = 'r';
    res['\t'] = 't';
    return res;
}();

// Lookup table used to determine "insignificant whitespace" characters when
// skipping whitespace during parser. Seems to be either similar or marginally
// faster in performance than a regular condition check.
constexpr std::array<bool, _number_of_char_values> _lookup_whitespace_chars = [] {
    std::array<bool, _number_of_char_values> res{};
    // "Insignificant whitespace" according to the JSON spec:
    // [https://ecma-international.org/wp-content/uploads/ECMA-404.pdf]
    // constitues following symbols:
    // - Whitespace      (aka ' ' )
    // - Tabs            (aka '\t')
    // - Carriage return (aka '\r')
    // - Newline         (aka '\n')
    res[' ']  = true;
    res['\t'] = true;
    res['\r'] = true;
    res['\n'] = true;
    return res;
}();

// Lookup table used to get an appropriate char for the escaped char in a 2-char JSON escape sequence.
// Allows us to avoid a chain of 8 if-else'es which ends up beings faster.
constexpr std::array<char, _number_of_char_values> _lookup_parsed_escaped_chars = [] {
    std::array<char, _number_of_char_values> res{};
    res['"']  = '"';
    res['\\'] = '\\';
    res['/']  = '/';
    res['b']  = '\b';
    res['f']  = '\f';
    res['n']  = '\n';
    res['r']  = '\r';
    res['t']  = '\t';
    return res;
}();

// Lookup table used to validate that JSON string has no unescaped charactes that should be escaped
constexpr std::array<bool, _number_of_char_values> _lookup_rejected_control_chars = [] {
    std::array<bool, _number_of_char_values> res{};
    // Codepoints U+0000 to U+001F require an escape
    // Some can be escaped with a 2-char sequence, others should be escaped as a unicode HEX
    for (char c = 0; c <= 31; ++c) res[c] = true;
    return res;

    // Note:
    // While C++ standard doesn't guarantee that chars 0-31 correspond to ASCII control characters,
    // this is in fact guaranteed by the assumption of UTF-8 encoded string.
}();

// ==========================
// --- JSON Parsing impl. ---
// ==========================

inline int _recursion_limit = 1000;

inline void set_recursion_limit(int max_depth) noexcept { _recursion_limit = max_depth; }

struct _parser {
    const std::string& chars;
    int                recursion_depth{};
    // we track recursion depth to handle stack allocation errors
    // (this can be caused malicious inputs with extreme level of nesting, for example, 100k array
    // opening brackets, which would cause huge recursion depth causing the stack to overflow with SIGSEGV)

    // dynamic allocation errors can be handled with regular exceptions through std::bad_alloc

    _parser() = delete;
    _parser(const std::string& chars) : chars(chars) {}

    // Parser state
    std::size_t skip_nonsignificant_whitespace(std::size_t cursor) {
        using namespace std::string_literals;

        while (cursor < this->chars.size()) {
            if (!_lookup_whitespace_chars[this->chars[cursor]]) return cursor;
            ++cursor;
        }

        throw std::runtime_error("JSON parser reached the end of buffer at pos "s + std::to_string(cursor) +
                                 " while skipping insignificant whitespace segment."s +
                                 _pretty_error(cursor, this->chars));
    }

    // Parsing methods
    std::pair<std::size_t, Node> parse_node(std::size_t cursor) {
        using namespace std::string_literals;

        // Node selector assumes it is starting at a significant symbol
        // which is the first symbol of the node to be parsed

        const char c = this->chars[cursor];

        // Assuming valid JSON, we can determine node type based on a single first character
        if (c == '{') {
            return this->parse_object(cursor);
        } else if (c == '[') {
            return this->parse_array(cursor);
        } else if (c == '"') {
            return this->parse_string(cursor);
        } else if (('0' <= c && c <= '9') || (c == '-')) {
            return this->parse_number(cursor);
        } else if (c == 't') {
            return this->parse_true(cursor);
        } else if (c == 'f') {
            return this->parse_false(cursor);
        } else if (c == 'n') {
            return this->parse_null(cursor);
        }
        throw std::runtime_error("JSON node selector encountered unexpected marker symbol {"s + this->chars[cursor] +
                                 "} at pos "s + std::to_string(cursor) + " (should be one of {0123456789{[\"tfn})."s +
                                 _pretty_error(cursor, this->chars));

        // Note: using a lookup table instead of an 'if' chain doesn't seem to offer any performance benefits here
    }

    std::size_t parse_object_pair(std::size_t cursor, Object& parent) {
        using namespace std::string_literals;

        // Object pair parser assumes it is starting at a '"'

        // Parse pair key
        std::string key; // allocating a string here is fine since we will std::move() into a map key
        std::tie(cursor, key) = this->parse_string(cursor); // may point 'this->current_node' to a new node

        // Handle stuff inbetween
        cursor = this->skip_nonsignificant_whitespace(cursor);
        if (this->chars[cursor] != ':')
            throw std::runtime_error("JSON object node encountered unexpected symbol {"s + this->chars[cursor] +
                                     "} after the pair key at pos "s + std::to_string(cursor) + " (should be {:})."s +
                                     _pretty_error(cursor, this->chars));
        ++cursor; // move past the colon ':'
        cursor = this->skip_nonsignificant_whitespace(cursor);

        // Parse pair value
        if (++this->recursion_depth > _recursion_limit)
            throw std::runtime_error("JSON parser has exceeded maximum allowed recursion depth of "s +
                                     std::to_string(_recursion_limit) +
                                     ". If stated depth wasn't caused by an invalid input, "s +
                                     "recursion limit can be increased with json::set_recursion_limit()."s);

        Node value;
        std::tie(cursor, value) = this->parse_node(cursor);

        --this->recursion_depth;

        // Note 1:
        // The question of wheter JSON allows duplicate keys is non-trivial but the resulting answer is NO.
        // JSON is goverened by 2 standards:
        // 1) ECMA-404 https://ecma-international.org/wp-content/uploads/ECMA-404.pdf
        //    which doesn't say anything about duplicate kys
        // 2) RFC-8259 https://www.rfc-editor.org/rfc/rfc2119
        //    which states "The names within an object SHOULD be unique.",
        //    however as defined in RFC-2119 https://www.rfc-editor.org/rfc/rfc2119:
        //       "SHOULD This word, or the adjective "RECOMMENDED", mean that there may exist valid reasons in
        //       particular circumstances to ignore a particular item, but the full implications must be understood
        //       and carefully weighed before choosing a different course."
        // which means at the end of the day duplicate keys are discouraged but still valid

        // Note 2:
        // There is no standard specification on which JSON value should be prefered in case of duplicate keys.
        // This is considered implementation detail as per RFC-8259:
        //    "An object whose names are all unique is interoperable in the sense that all software implementations
        //    receiving that object will agree on the name-value mappings. When the names within an object are not
        //    unique, the behavior of software that receives such an object is unpredictable. Many implementations
        //    report the last name/value pair only. Other implementations report an error or fail to parse the object,
        //    and some implementations report all of the name/value pairs, including duplicates."

        // Note 3:
        // We could easily check for duplicate keys since 'std::map<>::emplace()' returns insertion success as a bool
        // (the same isn't true for 'std::map<>::emplace_hint()' which returns just the iterator), however we will
        // not since that goes against the standard

        // Note 4:
        // 'parent.emplace_hint(parent.end(), ...)' can drastically speed up parsing of sorted JSON object, however
        // since most JSONs in the wild aren't sorted we will resort to a more generic option of regular '.emplace()'

        parent.emplace(std::move(key), std::move(value));

        return cursor;
    }

    std::pair<std::size_t, Object> parse_object(std::size_t cursor) {
        using namespace std::string_literals;

        ++cursor; // move past the opening brace '{'

        // Empty object that will accumulate child nodes as we parse them
        Object object_value;

        // Handle 1st pair
        cursor = this->skip_nonsignificant_whitespace(cursor);
        if (this->chars[cursor] != '}') {
            cursor = this->parse_object_pair(cursor, object_value);
        } else {
            ++cursor; // move past the closing brace '}'
            return {cursor, std::move(object_value)};
        }

        // Handle other pairs

        // Since we are staring past the first pair, all following pairs are gonna be preceded by a comma.
        //
        // Strictly speaking, commas in objects aren't necessary for decoding a JSON, this is
        // a case of redundant information, included into the format to make it more human-readable.
        // { "key_1":1 "key_1":2 "key_3":"value" } <- enough information to parse without commas.
        //
        // However commas ARE in fact necessary for array parsing. By using commas to detect when we have
        // to parse another pair, we can reuse the same algorithm for both objects pairs and array elements.
        //
        // Doing so also has a benefit of inherently adding comma-presense validation which we would have
        // to do manually otherwise.

        while (cursor < this->chars.size()) {
            cursor       = this->skip_nonsignificant_whitespace(cursor);
            const char c = this->chars[cursor];

            if (c == ',') {
                ++cursor; // move past the comma ','
                cursor = this->skip_nonsignificant_whitespace(cursor);
                cursor = this->parse_object_pair(cursor, object_value);
            } else if (c == '}') {
                ++cursor; // move past the closing brace '}'
                return {cursor, std::move(object_value)};
            } else {
                throw std::runtime_error(
                    "JSON array node could not find comma {,} or object ending symbol {}} after the element at pos "s +
                    std::to_string(cursor) + "."s + _pretty_error(cursor, this->chars));
            }
        }

        throw std::runtime_error("JSON object node reached the end of buffer while parsing object contents." +
                                 _pretty_error(cursor, this->chars));
    }

    std::size_t parse_array_element(std::size_t cursor, Array& parent) {
        using namespace std::string_literals;

        // Array element parser assumes it is starting at the first symbol of some JSON node

        // Parse pair key
        if (++this->recursion_depth > _recursion_limit)
            throw std::runtime_error("JSON parser has exceeded maximum allowed recursion depth of "s +
                                     std::to_string(_recursion_limit) +
                                     ". If stated depth wasn't caused by an invalid input, "s +
                                     "recursion limit can be increased with json::set_recursion_limit()."s);

        Node value;
        std::tie(cursor, value) = this->parse_node(cursor);

        --this->recursion_depth;

        parent.emplace_back(std::move(value));

        return cursor;
    }

    std::pair<std::size_t, Array> parse_array(std::size_t cursor) {
        using namespace std::string_literals;

        ++cursor; // move past the opening bracket '['

        // Empty object that will accumulate child nodes as we parse them
        Array array_value;

        // Handle 1st pair
        cursor = this->skip_nonsignificant_whitespace(cursor);
        if (this->chars[cursor] != ']') {
            cursor = this->parse_array_element(cursor, array_value);
        } else {
            ++cursor; // move past the closing bracket ']'
            return {cursor, std::move(array_value)};
        }

        // Handle other pairs
        // (the exact same way we do with objects, see the note here)
        while (cursor < this->chars.size()) {
            cursor       = this->skip_nonsignificant_whitespace(cursor);
            const char c = this->chars[cursor];

            if (c == ',') {
                ++cursor; // move past the comma ','
                cursor = this->skip_nonsignificant_whitespace(cursor);
                cursor = this->parse_array_element(cursor, array_value);
            } else if (c == ']') {
                ++cursor; // move past the closing bracket ']'
                return {cursor, std::move(array_value)};
            } else {
                throw std::runtime_error(
                    "JSON array node could not find comma {,} or array ending symbol {]} after the element at pos "s +
                    std::to_string(cursor) + "."s + _pretty_error(cursor, this->chars));
            }
        }

        throw std::runtime_error("JSON array node reached the end of buffer while parsing object contents." +
                                 _pretty_error(cursor, this->chars));
    }

    std::pair<std::size_t, String> parse_string(std::size_t cursor) {
        using namespace std::string_literals;

        // Empty string that will accumulate characters as we parse them
        std::string string_value;

        ++cursor; // move past the opening quote '"'

        // Serialize string while handling escape sequences.
        //
        // Doing 'string_value += c' for every char is ~50-60% slower than appending whole string at once,
        // which is why we 'buffer' appends by keeping track of 'segment_start' and 'cursor', and appending
        // whole chunks of the buffer to 'string_value' when we encounter an escape sequence or end of the string.
        //
        for (std::size_t segment_start = cursor; cursor < this->chars.size(); ++cursor) {
            const char c = this->chars[cursor];

            // Handle escape sequences inside the string
            if (c == '\\') {
                ++cursor; // move past the backslash '\'

                string_value.append(this->chars.data() + segment_start, cursor - segment_start - 1);
                // can't buffer more than that since we have to insert special characters now.

                const char escaped_char = this->chars[cursor];

                // 2-character escape sequences
                if (const char replacement_char = _lookup_parsed_escaped_chars[escaped_char]) {
                    if (cursor >= this->chars.size())
                        throw std::runtime_error("JSON string node reached the end of buffer while"s +
                                                 "parsing a 2-character escape sequence at pos "s +
                                                 std::to_string(cursor) + "."s + _pretty_error(cursor, this->chars));
                    string_value += replacement_char;
                }
                // 6-character escape sequences (escaped unicode HEX codepoints)
                else if (escaped_char == 'u') {
                    if (cursor >= this->chars.size() + 4)
                        throw std::runtime_error("JSON string node reached the end of buffer while"s +
                                                 "parsing a 5-character escape sequence at pos "s +
                                                 std::to_string(cursor) + "."s + _pretty_error(cursor, this->chars));

                    // Standard library is absolutely HORRIBLE when it comes to Unicode support.
                    // Literally every single encoding function in <cuchar>/<string>/<codecvt> is a
                    // crime against common sense, API safety and performace, which is why we do this
                    // inefficient nonsense and pray that there's not gonna be a lot of escaped unicode
                    // in the data. This also adds another 2 #include's just by itself. Supports UTF-8.
                    const std::string hex(this->chars.data() + cursor + 1, 4);
                    // say hello to allocation, standard functions only support std::string or null-terminated char*,
                    // there's no way (that I know of) to view into data and parse it without copying it
                    const char32_t    unicode_char = std::stoul(hex, nullptr, 16);
                    if (!_unicode_codepoint_to_utf8(string_value, unicode_char))
                        throw std::runtime_error("JSON string node could not parse unicode codepoint {"s + hex +
                                                 "} while parsing an escape sequence at pos "s +
                                                 std::to_string(cursor) + "."s + _pretty_error(cursor, this->chars));
                    cursor += 4; // move past first 'uXXX' symbols, last symbol will be covered by the loop '++cursor'
                } else {
                    throw std::runtime_error("JSON string node encountered unexpected character {"s + escaped_char +
                                             "} while parsing an escape sequence at pos "s + std::to_string(cursor) +
                                             "."s + _pretty_error(cursor, this->chars));
                }

                // This covers all non-hex escape sequences according to ECMA-404 specification
                // [https://ecma-international.org/wp-content/uploads/ECMA-404.pdf] (page 4)

                // moving past the escaped character will be done by the loop '++cursor'
                segment_start = cursor + 1;
                continue;
            }
            // validation
            else if (_lookup_rejected_control_chars[c])
                throw std::runtime_error(
                    "JSON string node encountered unescaped ASCII control character character \\"s +
                    std::to_string(static_cast<int>(c)) + " at pos "s + std::to_string(cursor) + "."s +
                    _pretty_error(cursor, this->chars));

            // Reached the end of the string
            if (c == '"') {
                string_value.append(this->chars.data() + segment_start, cursor - segment_start);
                ++cursor; // move past the closing quote '"'
                return {cursor, std::move(string_value)};
            }
        }

        throw std::runtime_error("JSON string node reached the end of buffer while parsing string contents." +
                                 _pretty_error(cursor, this->chars));
    }

    std::pair<std::size_t, Number> parse_number(std::size_t cursor) {
        using namespace std::string_literals;

        Number number_value;

        const auto [numer_end_ptr, error_code] =
            std::from_chars(this->chars.data() + cursor, this->chars.data() + this->chars.size(), number_value);

        // Note:
        // std::from_chars() converts the first complete number it finds in the string,
        // for example "42 meters" would be converted to 42. We rely on that behaviour here.

        if (error_code != std::errc()) {
            // std::errc(0) is a valid enumeration value that represents success
            // even though it does not appear in the enumerator list (which starts at 1)
            if (error_code == std::errc::invalid_argument)
                throw std::runtime_error("JSON number node could not be parsed as a number at pos "s +
                                         std::to_string(cursor) + "."s + _pretty_error(cursor, this->chars));
            else if (error_code == std::errc::result_out_of_range)
                throw std::runtime_error(
                    "JSON number node parsed to number larger than its possible binary representation at pos "s +
                    std::to_string(cursor) + "."s + _pretty_error(cursor, this->chars));
        }

        return {numer_end_ptr - this->chars.data(), number_value};
    }

    std::pair<std::size_t, Bool> parse_true(std::size_t cursor) {
        using namespace std::string_literals;
        constexpr std::size_t token_length = 4;

        if (cursor + token_length > this->chars.size())
            throw std::runtime_error("JSON bool node reached the end of buffer while parsing {true}." +
                                     _pretty_error(cursor, this->chars));

        const bool parsed_correctly =         //
            this->chars[cursor + 0] == 't' && //
            this->chars[cursor + 1] == 'r' && //
            this->chars[cursor + 2] == 'u' && //
            this->chars[cursor + 3] == 'e';   //

        if (!parsed_correctly)
            throw std::runtime_error("JSON bool node could not parse {true} at pos "s + std::to_string(cursor) + "."s +
                                     _pretty_error(cursor, this->chars));

        return {cursor + token_length, Bool(true)};
    }

    std::pair<std::size_t, Bool> parse_false(std::size_t cursor) {
        using namespace std::string_literals;
        constexpr std::size_t token_length = 5;

        if (cursor + token_length > this->chars.size())
            throw std::runtime_error("JSON bool node reached the end of buffer while parsing {false}." +
                                     _pretty_error(cursor, this->chars));

        const bool parsed_correctly =         //
            this->chars[cursor + 0] == 'f' && //
            this->chars[cursor + 1] == 'a' && //
            this->chars[cursor + 2] == 'l' && //
            this->chars[cursor + 3] == 's' && //
            this->chars[cursor + 4] == 'e';   //

        if (!parsed_correctly)
            throw std::runtime_error("JSON bool node could not parse {false} at pos "s + std::to_string(cursor) + "."s +
                                     _pretty_error(cursor, this->chars));

        return {cursor + token_length, Bool(false)};
    }

    std::pair<std::size_t, Null> parse_null(std::size_t cursor) {
        using namespace std::string_literals;
        constexpr std::size_t token_length = 4;

        if (cursor + token_length > this->chars.size())
            throw std::runtime_error("JSON null node reached the end of buffer while parsing {null}." +
                                     _pretty_error(cursor, this->chars));

        const bool parsed_correctly =         //
            this->chars[cursor + 0] == 'n' && //
            this->chars[cursor + 1] == 'u' && //
            this->chars[cursor + 2] == 'l' && //
            this->chars[cursor + 3] == 'l';   //

        if (!parsed_correctly)
            throw std::runtime_error("JSON null node could not parse {null} at pos "s + std::to_string(cursor) + "."s +
                                     _pretty_error(cursor, this->chars));

        return {cursor + token_length, Null()};
    }
};


// ==============================
// --- JSON Serializing impl. ---
// ==============================

template <bool prettify>
inline void _serialize_json_recursion(const Node& node, std::string& chars, unsigned int indent_level = 0,
                                      bool skip_first_indent = false) {
    using namespace std::string_literals;
    constexpr std::size_t indent_level_size = 4;
    const std::size_t     indent_size       = indent_level_size * indent_level;

    // first indent should be skipped when printing after a key
    //
    // Example:
    //
    // {
    //     "object": {              <- first indent skipped (Object)
    //         "something": null    <- first indent skipped (Null)
    //     },
    //     "array": [               <- first indent skipped (Array)
    //          1,                  <- first indent NOT skipped (Number)
    //          2                   <- first indent NOT skipped (Number)
    //     ]
    // }
    //
    // We handle 'prettify' segments through 'if constexpr'
    // to avoid  any "trace" overhead on non-prettified serializing
    //

    // Note:
    // The fastest way to append strings to a preallocated buffer seems to be with '+=':
    //    > chars += string_1; chars += string_2; chars += string_3;
    //
    // Using operator '+' slows things down due to additional allocations:
    //    > chars +=  string_1 + string_2 + string_3; // slow
    //
    // '.append()' performs exactly the same as '+=', but has no overload for appending single chars.
    // However it does have an overload for appending N of some character, which is why we use if for indentation.
    //
    // 'std::ostringstream' is painfully slow compared to regular appends
    // so it's out of the question.

    if constexpr (prettify)
        if (!skip_first_indent) chars.append(indent_size, ' ');

    // JSON Object
    if (auto* ptr = node.get_if<Object>()) {
        const auto& object_value = *ptr;

        // Skip all logic for empty objects
        if (object_value.empty()) {
            chars += "{}";
            return;
        }

        chars += '{';
        if constexpr (prettify) chars += '\n';

        for (auto it = object_value.cbegin();;) {
            if constexpr (prettify) chars.append(indent_size + indent_level_size, ' ');
            // Key
            chars += '"';
            chars += it->first;
            if constexpr (prettify) chars += "\": ";
            else chars += "\":";
            // Value
            _serialize_json_recursion<prettify>(it->second, chars, indent_level + 1, true);
            // Comma
            if (++it != object_value.cend()) { // prevents trailing comma
                chars += ',';
                if constexpr (prettify) chars += '\n';
            } else {
                if constexpr (prettify) chars += '\n';
                break;
            }
        }

        if constexpr (prettify) chars.append(indent_size, ' ');
        chars += '}';
    }
    // JSON Array
    else if (auto* ptr = node.get_if<Array>()) {
        const auto& array_value = *ptr;

        // Skip all logic for empty arrays
        if (array_value.empty()) {
            chars += "[]";
            return;
        }

        chars += '[';
        if constexpr (prettify) chars += '\n';

        for (auto it = array_value.cbegin();;) {
            // Node
            _serialize_json_recursion<prettify>(*it, chars, indent_level + 1);
            // Comma
            if (++it != array_value.cend()) { // prevents trailing comma
                chars += ',';
                if constexpr (prettify) chars += '\n';
            } else {
                if constexpr (prettify) chars += '\n';
                break;
            }
        }
        if constexpr (prettify) chars.append(indent_size, ' ');
        chars += ']';
    }
    // String
    else if (auto* ptr = node.get_if<String>()) {
        const auto& string_value = *ptr;

        chars += '"';

        // Serialize string while handling escape sequences.
        /// Without escape sequences we could just do 'chars += string_value'.
        //
        // Since appending individual characters is ~twice as slow as appending the whole string, we use a
        // "buffered" way of appending, appending whole segments up to the currently escaped char.
        // Strings with no escaped chars get appended in a single call.
        //
        std::size_t segment_start = 0;
        for (std::size_t i = 0; i < string_value.size(); ++i) {
            if (const char escaped_char_replacement = _lookup_serialized_escaped_chars[string_value[i]]) {
                chars.append(string_value.data() + segment_start, i - segment_start);
                chars += '\\';
                chars += escaped_char_replacement;
                segment_start = i + 1; // skip over the "actual" technical character in the string
            }
        }
        chars.append(string_value.data() + segment_start, string_value.size() - segment_start);

        chars += '"';
    }
    // Number
    else if (auto* ptr = node.get_if<Number>()) {
        const auto& number_value = *ptr;

        constexpr int max_exponent = std::numeric_limits<Number>::max_exponent10;
        constexpr int max_digits =
            4 + std::numeric_limits<Number>::max_digits10 + std::max(2, _log_10_ceil(max_exponent));
        // should be the smallest buffer size to account for all possible 'std::to_chars()' outputs,
        // see [https://stackoverflow.com/questions/68472720/stdto-chars-minimal-floating-point-buffer-size]

        std::array<char, max_digits> buffer;

        const auto [number_end_ptr, error_code] =
            std::to_chars(buffer.data(), buffer.data() + buffer.size(), number_value);

        if (error_code != std::errc())
            throw std::runtime_error(
                "JSON serializing encountered std::to_chars() formatting error while serializing value {"s +
                std::to_string(number_value) + "}."s);

        const std::string_view number_string(buffer.data(), number_end_ptr - buffer.data());

        // Save NaN/Inf cases as strings, since JSON spec doesn't include IEEE 754.
        // (!) May result in non-homogenous arrays like [ 1.0, "inf" , 3.0, 4.0, "nan" ]
        if (std::isfinite(number_value)) {
            chars.append(buffer.data(), number_end_ptr - buffer.data());
        } else {
            chars += '"';
            chars.append(buffer.data(), number_end_ptr - buffer.data());
            chars += '"';
        }
    }
    // Bool
    else if (auto* ptr = node.get_if<Bool>()) {
        const auto& bool_value = *ptr;
        chars += (bool_value ? "true" : "false");
    }
    // Null
    else if (node.is<Null>()) {
        chars += "null";
    }
}

inline void _serialize_json_to_buffer(std::string& chars, const Node& node, Format format) {
    if (format == Format::PRETTY) _serialize_json_recursion<true>(node, chars);
    else _serialize_json_recursion<false>(node, chars);
}

// ===============================
// --- JSON Parsing public API ---
// ===============================

inline Node from_string(const std::string& chars) {
    _parser           parser(chars);
    const std::size_t json_start   = parser.skip_nonsignificant_whitespace(0); // skip leading whitespace
    auto [end_cursor, result_node] = parser.parse_node(json_start); // starts parsing recursively from the root node

    // Check for invalid trailing sumbols
    using namespace std::string_literals;
    for (auto cursor = end_cursor; cursor < chars.size(); ++cursor)
        if (!_lookup_whitespace_chars[chars[cursor]])
            throw std::runtime_error("Invalid trailing symbols encountered after the root JSON node at pos "s +
                                     std::to_string(cursor) + "."s + _pretty_error(cursor, chars));

    return result_node;
}
inline Node from_file(const std::string& filepath) {
    const std::string chars = _read_file_to_string(filepath);
    return from_string(chars);
}

namespace literals {
inline Node operator""_utl_json(const char* c_str, std::size_t c_str_size) {
    return from_string(std::string(c_str, c_str_size));
}
} // namespace literals

} // namespace utl::json

#endif
#endif // module utl::json





// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DmitriBogdanov/prototyping_utils ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Module:        utl::log
// Documentation: https://github.com/DmitriBogdanov/prototyping_utils/blob/master/docs/module_log.md
// Source repo:   https://github.com/DmitriBogdanov/prototyping_utils
//
// This project is licensed under the MIT License
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#include <cstddef>

#include <sstream>
#include <string>
#include <utility>


#if !defined(UTL_PICK_MODULES) || defined(UTLMODULE_LOG)
#ifndef UTLHEADERGUARD_LOG
#define UTLHEADERGUARD_LOG

// _______________________ INCLUDES _______________________

#include <array>         // array<>
#include <charconv>      // to_chars()
#include <chrono>        // steady_clock
#include <fstream>       // ofstream
#include <iostream>      // cout
#include <list>          // list<>
#include <mutex>         // lock_guard<>, mutex
#include <ostream>       // ostream
#include <stdexcept>     // std::runtime_error
#include <string_view>   // string_view
#include <system_error>  // errc()
#include <thread>        // this_thread::get_id()
#include <type_traits>   // is_integral_v<>, is_floating_point_v<>, is_same_v<>, is_convertible_to_v<>
#include <unordered_map> // unordered_map<>
#include <vector>        // vector<>

// ____________________ DEVELOPER DOCS ____________________

// NOTE: DOCS

// ____________________ IMPLEMENTATION ____________________

namespace utl::log {

// ======================
// --- Internal utils ---
// ======================

// - SFINAE to select localtime_s() or localtime_r() -
template <typename Arg_tm, typename Arg_time_t>
auto _available_localtime_impl(Arg_tm time_moment, Arg_time_t timer)
    -> decltype(localtime_s(std::forward<Arg_tm>(time_moment), std::forward<Arg_time_t>(timer))) {
    return localtime_s(std::forward<Arg_tm>(time_moment), std::forward<Arg_time_t>(timer));
}

template <typename Arg_tm, typename Arg_time_t>
auto _available_localtime_impl(Arg_tm time_moment, Arg_time_t timer)
    -> decltype(localtime_r(std::forward<Arg_time_t>(timer), std::forward<Arg_tm>(time_moment))) {
    return localtime_r(std::forward<Arg_time_t>(timer), std::forward<Arg_tm>(time_moment));
}

std::size_t _get_thread_index(const std::thread::id id) {
    static std::size_t next_index = 0;

    static std::mutex                                       mutex;
    static std::unordered_map<std::thread::id, std::size_t> thread_ids;
    std::lock_guard<std::mutex>                             lock(mutex);

    const auto it = thread_ids.find(id);
    if (it == thread_ids.end()) return thread_ids[id] = next_index++;
    return it->second;
}

template <class IntType, std::enable_if_t<std::is_integral<IntType>::value, bool> = true>
unsigned int _integer_digit_count(IntType value) {
    unsigned int digits = (value <= 0) ? 1 : 0;
    // (value <  0) => we add 1 digit because of '-' in front
    // (value == 0) => we add 1 digit for '0' because the loop doesn't account for zero integers
    while (value) {
        value /= 10;
        ++digits;
    }
    return digits;
    // Note: There is probably a faster way of doing it
}

using clock = std::chrono::steady_clock;

using ms = std::chrono::microseconds;

inline const clock::time_point _program_entry_time_point = clock::now();

// =======================
// --- Stringification ---
// =======================

// --- Internal type traits ---
// ----------------------------

#define utl_log_define_trait(trait_name_, ...)                                                                         \
    template <class T, class = void>                                                                                   \
    struct trait_name_ : std::false_type {};                                                                           \
                                                                                                                       \
    template <class T>                                                                                                 \
    struct trait_name_<T, std::void_t<decltype(__VA_ARGS__)>> : std::true_type {};                                     \
                                                                                                                       \
    template <class T>                                                                                                 \
    constexpr bool trait_name_##_v = trait_name_<T>::value;

utl_log_define_trait(_has_string_append, std::string() += std::declval<T>());
utl_log_define_trait(_has_begin, ++std::declval<T>().begin());
utl_log_define_trait(_has_end, ++std::declval<T>().end());
utl_log_define_trait(_has_get, std::get<0>(std::declval<T>()));
utl_log_define_trait(_has_tuple_size, std::tuple_size<T>::value);
utl_log_define_trait(_has_ostream_insert, std::declval<std::ostream>() << std::declval<T>());

#undef utl_log_define_trait

// --- Internal utils ---
// ----------------------

template <class>
inline constexpr bool _always_false_v = false;

template <typename T>
constexpr int _log_10_ceil(T num) {
    return num < 10 ? 1 : 1 + _log_10_ceil(num / 10);
}

template <typename T>
constexpr std::size_t _max_float_digits =
    4 + std::numeric_limits<T>::max_digits10 + std::max(2, _log_10_ceil(std::numeric_limits<T>::max_exponent10));
// should be the smallest buffer size to account for all possible 'std::to_chars()' outputs,
// see [https://stackoverflow.com/questions/68472720/stdto-chars-minimal-floating-point-buffer-size]

// --- Stringifiers ---
// --------------------

template <class T>
void append_stringified(std::string& str, const T& value);

void _append_stringified_bool(std::string& str, bool value) { str += value ? "true" : "false"; }

// Fast implementation for stringifying an integer and appending it to 'std::string'
template <class T>
void _append_stringified_integer(std::string& str, T value) {

    // Note:
    // We could count the digits of 'value', preallocate buffer for exactly however many characters
    // we need and format directly to it, however benchmarks showed that it is actually inferior to
    // just doing things the usual way with a stack-allocated middle-man buffer.
    std::array<char, std::numeric_limits<T>::digits10> buffer;
    const auto [number_end_ptr, error_code] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);

    if (error_code != std::errc())
        throw std::runtime_error(
            "Integer stringification encountered std::to_chars() formatting error while serializing a value.");

    str.append(buffer.data(), number_end_ptr - buffer.data());
}

// Fast implementation for stringifying a float and appending it to 'std::string'
template <class T>
void _append_stringified_float(std::string& str, T value) {
    std::array<char, _max_float_digits<T>> buffer;
    const auto [number_end_ptr, error_code] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);

    if (error_code != std::errc())
        throw std::runtime_error(
            "Float stringification encountered std::to_chars() formatting error while serializing a value.");

    str.append(buffer.data(), number_end_ptr - buffer.data());
}

template <class T>
void _append_stringified_stringlike(std::string& str, const T& value) {
    str += value;
    // 'std::string' has operator '+=' for following types:
    // 1) 'const std::string&'
    // 2) char-like (includes all integral types)
    // 3) 'std::string_view'-convertible
}

template <class T>
void _append_stringified_string_convertible(std::string& str, const T& value) {
    str += std::string(value);
}

template <class T>
void _append_stringified_array(std::string& str, const T& value) {
    str += "{ ";
    if (value.begin() != value.end())
        for (auto it = value.begin();;) {
            append_stringified(str, *it);
            if (++it == value.end()) break; // prevents trailing comma
            str += ", ";
        }
    str += " }";
}

template <class Tuplelike, std::size_t... Idx>
void _append_stringified_tuple_impl(std::string& str, Tuplelike value, std::index_sequence<Idx...>) {
    ((Idx == 0 ? "" : str += ", ", append_stringified(str, std::get<Idx>(value))), ...);
    // fold expression '( f(args), ... )' invokes 'f(args)' for all arguments in 'args...'
    // in the same fashion, we can fold over 2 functions by doing '( ( f(args), g(args) ), ... )'
}

template <template <class...> class Tuplelike, class... Args>
void _append_stringified_tuple(std::string& str, const Tuplelike<Args...>& value) {
    str += "< ";
    _append_stringified_tuple_impl(str, value, std::index_sequence_for<Args...>{});
    str += " >";
}

template <class T>
void _append_stringified_printable(std::string& str, const T& value) {
    str += (std::ostringstream() << value).str(); // rather slow but at this point there is no other option
}

// --- Public API ---
// ------------------

template <class T>
void append_stringified(std::string& str, const T& value) {
    // Bool
    if constexpr (std::is_same_v<T, bool>) _append_stringified_bool(str, value);
    // Char
    else if constexpr (std::is_same_v<T, char>) _append_stringified_stringlike(str, value);
    // Integral
    else if constexpr (std::is_integral_v<T>) _append_stringified_integer(str, value);
    // Floating-point
    else if constexpr (std::is_floating_point_v<T>) _append_stringified_float(str, value);
    // 'std::string_view'-convertible (most strings and string-like types)
    else if constexpr (std::is_convertible_v<T, std::string_view>) _append_stringified_stringlike(str, value);
    // 'std::string'-convertible (some "nastier" string-like types, mainly 'std::path')
    else if constexpr (std::is_convertible_v<T, std::string>) _append_stringified_string_convertible(str, value);
    // Array-like
    else if constexpr (_has_begin_v<T> && _has_end_v<T>) _append_stringified_array(str, value);
    // Tuple-like
    else if constexpr (_has_get_v<T> && _has_tuple_size_v<T>) _append_stringified_tuple(str, value);
    // 'std::ostream' printable
    else if constexpr (_has_ostream_insert_v<T>) _append_stringified_printable(str, value);
    // No valid stringification exists
    else static_assert(_always_false_v<T>, "No valid stringification exists for the type.");

    // Note: Using if-constexpr chain here allows us to pick and choose priority of different branches,
    // removing any possible ambiguity we could encounter doing things through SFINAE or overloads.
}

template <class... Args>
std::string stringify(Args&&... args) {
    std::string buffer;
    (append_stringified(buffer, std::forward<Args>(args)), ...);
    return buffer;
}

template <class... Args>
void print(Args&&... args) {
    std::cout << stringify(std::forward<Args>(args)...);
}

template <class... Args>
void println(Args&&... args) {
    std::cout << stringify(std::forward<Args>(args)...) << '\n';
}

// ===============
// --- Options ---
// ===============

enum class Verbosity { ERR = 1, WARN = 2, INFO = 3, TRACE = 4 };

enum class OpenMode { REWRITE, APPEND };

enum class Colors { ENABLE, DISABLE };

struct Columns {
    bool datetime;
    bool uptime;
    bool thread;
    bool callsite;
    bool level;
    bool message;

    Columns() : datetime(true), uptime(true), thread(true), callsite(true), level(true), message(true) {}
};

struct Callsite {
    std::string_view file;
    int              line;
};

struct MessageMetadata {
    Verbosity verbosity;
};

constexpr bool operator<(Verbosity l, Verbosity r) { return static_cast<int>(l) < static_cast<int>(r); }
constexpr bool operator<=(Verbosity l, Verbosity r) { return static_cast<int>(l) <= static_cast<int>(r); }

// ==================
// --- Sink class ---
// ==================

class Sink {
private:
    std::ostream&     os;
    Verbosity         verbosity;
    Colors            colors;
    clock::duration   flush_interval;
    Columns           columns;
    clock::time_point last_flushed;

public:
    Sink()            = delete;
    Sink(const Sink&) = delete;
    Sink(Sink&&)      = default;

    Sink(std::ostream& os, Verbosity verbosity, Colors colors, clock::duration flush_interval, const Columns& columns)
        : os(os), verbosity(verbosity), colors(colors), flush_interval(flush_interval), columns(columns) {}

    template <typename... Args>
    void format(const Callsite& callsite, const MessageMetadata& meta, const Args&... args) {
        if (meta.verbosity > this->verbosity) return;

        thread_local std::string buffer;

        const clock::time_point now = clock::now();

        // To minimize logging overhead we use string buffer, append characters to it and then write the whole buffer
        // to `std::ostream`. This avoids the inherent overhead of ostream formatting (caused largely by
        // virtualization, syncronization and locale handling, neither of which are relevant for the logger).
        //
        // This buffer gets reused between calls. Note the 'thread_local', if buffer was just a class member, multiple
        // threads could fight trying to clear and resize the buffer while it's being written to by another thread.
        //
        // Buffer may grow when formatting a message longer that any one that was formatted before, otherwise we just
        // reuse the reserved memory and no new allocations take place.

        buffer.clear();

        // Format columns one-by-one
        if (this->colors == Colors::ENABLE) switch (meta.verbosity) {
            case Verbosity::ERR: buffer += "\033[31;1m"; break;
            case Verbosity::WARN: buffer += "\033[33m"; break;
            case Verbosity::INFO: buffer += "\033[37m"; break;
            case Verbosity::TRACE: buffer += "\033[90m"; break;
            }

        if (this->columns.datetime) this->format_column_datetime(buffer);
        if (this->columns.uptime) this->format_column_uptime(buffer, now);
        if (this->columns.thread) this->format_column_thread(buffer);
        if (this->columns.callsite) this->format_column_callsite(buffer, callsite);
        if (this->columns.level) this->format_column_level(buffer, meta.verbosity);
        if (this->columns.message) this->format_column_message(buffer, args...);

        if (this->colors == Colors::ENABLE) { buffer += "\033[0m"; }

        buffer += '\n';

        this->os.write(buffer.data(), buffer.size());

        // flush every message immediately
        if (this->flush_interval.count() == 0) {
            this->os.flush();
            return;
        }
        // or flush periodically
        if (now - this->last_flushed > this->flush_interval) {
            this->last_flushed = now;
            this->os.flush();
        }
    }

    void format_column_datetime(std::string& buffer) {
        std::time_t timer = std::time(nullptr);
        std::tm     time_moment{};

        _available_localtime_impl(&time_moment, &timer);

        constexpr std::size_t datetime_width = sizeof("yyyy-mm-dd HH:MM:SS");
        // size includes the null terminator added by 'strftime()'

        // Format time straight into the buffer
        std::array<char, datetime_width> strftime_buffer;
        std::strftime(strftime_buffer.data(), strftime_buffer.size(), "%Y-%m-%d %H:%M:%S", &time_moment);

        strftime_buffer.back() = ' '; // replace null-terminator added by 'strftime()' with a space
        buffer.append(strftime_buffer.data(), strftime_buffer.size());
    }

    void format_column_uptime(std::string& buffer, clock::time_point now) {
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - _program_entry_time_point);
        const auto sec        = (elapsed_ms / 1000).count();
        const auto ms         = (elapsed_ms % 1000).count(); // is 'elapsed_ms - 1000 * full_seconds; faster?

        const unsigned int sec_digits = _integer_digit_count(sec);
        const unsigned int ms_digits  = _integer_digit_count(ms);

        constexpr unsigned int sec_width = 4;
        constexpr unsigned int ms_width  = 3;

        buffer += '(';

        // Emulate '<< std::right << std::setw(sec_width) << sec'
        if (sec_digits < sec_width) buffer.append(sec_width - sec_digits, ' ');
        append_stringified(buffer, sec);

        buffer += '.';

        // Emulate '<< std::right << std::setw(ms_width) << std::setfill('0') << ms'
        if (ms_digits < ms_width) buffer.append(ms_width - ms_digits, '0');
        append_stringified(buffer, ms);

        buffer += ')';
    }

    void format_column_thread(std::string& buffer) {
        buffer += '[';
        constexpr std::size_t thread_id_width = sizeof("thread") - 1;
        const auto            thread_id       = _get_thread_index(std::this_thread::get_id());
        if (_integer_digit_count(thread_id) < thread_id_width) buffer.append(thread_id_width - thread_id, ' ');
        append_stringified(buffer, thread_id);
        buffer += ']';
    }

    void format_column_callsite(std::string& buffer, const Callsite& callsite) {
        constexpr std::streamsize width_before_dot = 28;
        constexpr std::streamsize width_after_dot  = 4;

        // Get just filename from the full path
        std::string_view filename = callsite.file.substr(callsite.file.find_last_of("/\\") + 1);

        // Emulate '<< std::right << std::setw(width_before_dot) << filename'
        // trim first characters if it's too long
        if (filename.size() < width_before_dot) buffer.append(width_before_dot - filename.size(), ' ');
        else filename.remove_prefix(width_before_dot - filename.size());

        buffer += filename;
        buffer += ':';
        // Emulate '<< std::left << std::setw(width_after_dot) << line'
        append_stringified(buffer, callsite.line);
        buffer.append(width_after_dot - _integer_digit_count(callsite.line), ' ');
    }

    void format_column_level(std::string& buffer, Verbosity level) {
        switch (level) {
        case Verbosity::ERR: buffer += "   ERR|"; return;
        case Verbosity::WARN: buffer += "  WARN|"; return;
        case Verbosity::INFO: buffer += "  INFO|"; return;
        case Verbosity::TRACE: buffer += " TRACE|"; return;
        }
    }

    template <typename... Args>
    void format_column_message(std::string& buffer, const Args&... args) {
        buffer += ' ';
        (append_stringified(buffer, args), ...);
        //(buffer += ... += args); // parenthesis here are necessary
    }
};

// ====================
// --- Logger class ---
// ====================

class Logger {
private:
    inline static std::vector<Sink>        sinks;
    inline static std::list<std::ofstream> managed_files;
    // we don't want 'managed_files' to reallocate its elements at any point
    // since that would leave corresponding sinks with dangling references,
    // which is why list is used

public:
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    template <typename... Args>
    void push_message(const Callsite& callsite, const MessageMetadata& meta, const Args&... args) {
        // When no sinks were manually created, default sink-to-terminal takes over
        if (this->sinks.empty()) {
            static Sink default_sink(std::cout, Verbosity::TRACE, Colors::ENABLE, ms(0), Columns{});
            default_sink.format(callsite, meta, args...);
        }

        for (auto& sink : this->sinks) sink.format(callsite, meta, args...);
    }

    Sink& emplace_sink(std::ostream& os, Verbosity verbosity, Colors colors, clock::duration flush_interval,
                       const Columns& columns) {
        return this->sinks.emplace_back(os, verbosity, colors, flush_interval, columns);
    }

    std::ofstream& emplace_managed_file(const std::string& filename, OpenMode open_mode) {
        const auto ios_open_mode = (open_mode == OpenMode::APPEND) ? std::ios::out | std::ios::app : std::ios::out;
        return this->managed_files.emplace_back(filename, ios_open_mode);
    }
};

// =======================
// --- Sink public API ---
// =======================

Sink& add_terminal_sink(std::ostream& os, Verbosity verbosity = Verbosity::INFO, Colors colors = Colors::ENABLE,
                        clock::duration flush_interval = ms{}, const Columns& columns = Columns{}) {
    return Logger::instance().emplace_sink(os, verbosity, colors, flush_interval, columns);
}

Sink& add_file_sink(const std::string& filename, OpenMode open_mode = OpenMode::REWRITE,
                    Verbosity verbosity = Verbosity::TRACE, Colors colors = Colors::DISABLE,
                    clock::duration flush_interval = ms{15}, const Columns& columns = Columns{}) {
    auto& os = Logger::instance().emplace_managed_file(filename, open_mode);
    return Logger::instance().emplace_sink(os, verbosity, colors, flush_interval, columns);
}

// ======================
// --- Logging macros ---
// ======================

#define UTL_LOG_ERR(...)                                                                                               \
    utl::log::Logger::instance().push_message({__FILE__, __LINE__}, {utl::log::Verbosity::ERR}, __VA_ARGS__)

#define UTL_LOG_WARN(...)                                                                                              \
    utl::log::Logger::instance().push_message({__FILE__, __LINE__}, {utl::log::Verbosity::WARN}, __VA_ARGS__)

#define UTL_LOG_INFO(...)                                                                                              \
    utl::log::Logger::instance().push_message({__FILE__, __LINE__}, {utl::log::Verbosity::INFO}, __VA_ARGS__)

#define UTL_LOG_TRACE(...)                                                                                             \
    utl::log::Logger::instance().push_message({__FILE__, __LINE__}, {utl::log::Verbosity::TRACE}, __VA_ARGS__)

#ifdef _DEBUG
#define UTL_LOG_DERR(...) UTL_LOG_ERR(__VA_ARGS__)
#define UTL_LOG_DWARN(...) UTL_LOG_WARN(__VA_ARGS__)
#define UTL_LOG_DINFO(...) UTL_LOG_INFO(__VA_ARGS__)
#define UTL_LOG_DTRACE(...) UTL_LOG_TRACE(__VA_ARGS__)
#else
#define UTL_LOG_DERR(...)
#define UTL_LOG_DWARN(...)
#define UTL_LOG_DINFO(...)
#define UTL_LOG_DTRACE(...)
#endif


} // namespace utl::log

#endif
#endif // module utl::log






// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DmitriBogdanov/prototyping_utils ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Module:        utl::math
// Documentation: https://github.com/DmitriBogdanov/prototyping_utils/blob/master/docs/module_math.md
// Source repo:   https://github.com/DmitriBogdanov/prototyping_utils
//
// This project is licensed under the MIT License
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#if !defined(UTL_PICK_MODULES) || defined(UTLMODULE_MATH)
#ifndef UTLHEADERGUARD_MATH
#define UTLHEADERGUARD_MATH

// _______________________ INCLUDES _______________________

#include <cassert>     // assert()
#include <cstddef>     // size_t
#include <functional>  // function<>
#include <type_traits> // enable_if_t<>, void_t<>, is_floating_point<>, is_arithmetic<>,
                       // conditional_t<>, is_integral<>, true_type, false_type
#include <utility>     // declval<>()
#include <vector>      // vector<>

// ____________________ DEVELOPER DOCS ____________________

// Coordinate transformations, mathematical constants and technical helper functions.
// A bit of a mix-bag-of-everything, but in the end pretty useful.
//
// # ::PI, ::PI_TWO, ::PI_HALF, ::E, ::GOLDEN_RATION #
// Constants.
//
// # ::is_addable_with_itself<Type> #
// Integral constant, returns in "::value" whether Type supports 'operator()+' with itself.
//
// # ::is_multipliable_by_scalar<Type> #
// Integral constant, returns in "::value" whether Type supports 'operator()*' with double.
//
// # ::is_sized<Type> #
// Integral constant, returns in "::value" whether Type supports '.size()' method.
//
// # ::abs(), ::sign(), ::sqr(), ::cube(), ::midpoint(), deg_to_rad(), rad_to_deg() #
// Constexpr templated math functions, useful when writing expressions with a "textbook form" math.
//
// # ::uint_difference() #
// Returns abs(uint - uint) with respect to uint size and possible overflow.
//
// # ::linspace() #
// Tabulates [min, max] range with N evenly spaced points and returns it as a vector.
//
// # ::ssize() #
// Returns '.size()' of the argument casted to 'int'.
// Essentially a shortcut for verbose 'static_cast<int>(container.size())'.
//
// # ::ternary_branchless() #
// Branchless ternary operator. Slightly slower that regular ternary on most CPUs.
// Should not be used unless branchess qualifier is necessary (like in GPU computation).
//
// # ::ternary_bitselect() #
// Faster branchless ternary for integer types.
// If 2nd return is ommited, 0 is assumed, which allows for significant optimization.

// ____________________ IMPLEMENTATION ____________________

namespace utl::math {

// =================
// --- Constants ---
// =================

constexpr double PI           = 3.14159265358979323846;
constexpr double PI_TWO       = 2. * PI;
constexpr double PI_HALF      = 0.5 * PI;
constexpr double E            = 2.71828182845904523536;
constexpr double GOLDEN_RATIO = 1.6180339887498948482;

// ===================
// --- Type Traits ---
// ===================

template <typename Type, typename = void>
struct is_addable_with_itself : std::false_type {};

template <typename Type>
struct is_addable_with_itself<Type, std::void_t<decltype(std::declval<Type>() + std::declval<Type>())>
                              // perhaps check that resulting type is same as 'Type', but that can cause issues
                              // with classes like Eigen::MatrixXd that return "foldables" that convert back to
                              // objects in the end
                              > : std::true_type {};

template <typename Type, typename = void>
struct is_multipliable_by_scalar : std::false_type {};

template <typename Type>
struct is_multipliable_by_scalar<Type, std::void_t<decltype(std::declval<Type>() * std::declval<double>())>>
    : std::true_type {};

template <typename Type, typename = void>
struct is_sized : std::false_type {};

template <typename Type>
struct is_sized<Type, std::void_t<decltype(std::declval<Type>().size())>> : std::true_type {};

template <typename FuncType, typename Signature>
using is_function_with_signature = std::is_convertible<FuncType, std::function<Signature>>;

// ======================
// --- Math functions ---
// ======================

template <typename Type, std::enable_if_t<std::is_scalar<Type>::value, bool> = true>
[[nodiscard]] constexpr Type abs(Type x) {
    return (x > Type(0)) ? x : -x;
}

template <typename Type, std::enable_if_t<std::is_scalar<Type>::value, bool> = true>
[[nodiscard]] constexpr Type sign(Type x) {
    return (x > Type(0)) ? Type(1) : Type(-1);
}

template <typename Type, std::enable_if_t<std::is_arithmetic<Type>::value, bool> = true>
[[nodiscard]] constexpr Type sqr(Type x) {
    return x * x;
}

template <typename Type, std::enable_if_t<std::is_arithmetic<Type>::value, bool> = true>
[[nodiscard]] constexpr Type cube(Type x) {
    return x * x * x;
}

template <typename Type, std::enable_if_t<utl::math::is_addable_with_itself<Type>::value, bool> = true,
          std::enable_if_t<utl::math::is_multipliable_by_scalar<Type>::value, bool> = true>
[[nodiscard]] constexpr Type midpoint(Type a, Type b) {
    return (a + b) * 0.5;
}

template <typename IntegerType, std::enable_if_t<std::is_integral<IntegerType>::value, bool> = true>
[[nodiscard]] constexpr int kronecker_delta(IntegerType i, IntegerType j) {
    // 'IntegerType' here is necessary to prevent enforcing static_cast<int>(...) on the callsite
    return (i == j) ? 1 : 0;
}

template <typename IntegerType, std::enable_if_t<std::is_integral<IntegerType>::value, bool> = true>
[[nodiscard]] constexpr int power_of_minus_one(IntegerType power) {
    return (power % IntegerType(2)) ? -1 : 1; // is there a faster way of doing it?
}


// --- deg/rad conversion ---
template <typename FloatType, std::enable_if_t<std::is_floating_point<FloatType>::value, bool> = true>
[[nodiscard]] constexpr FloatType deg_to_rad(FloatType degrees) {
    constexpr FloatType FACTOR = FloatType(PI / 180.);
    return degrees * FACTOR;
}

template <typename FloatType, std::enable_if_t<std::is_floating_point<FloatType>::value, bool> = true>
[[nodiscard]] constexpr FloatType rad_to_deg(FloatType radians) {
    constexpr FloatType FACTOR = FloatType(180. / PI);
    return radians * FACTOR;
}

// ====================
// --- Memory Units ---
// ====================

// Workaround for 'static_assert(false)' making program ill-formed even
// when placed inide an 'if constexpr' branch that never compiles.
// 'static_assert(_always_false_v<T>)' on the the other hand doesn't,
// which means we can use it to mark branches that should never compile.
template <class>
inline constexpr bool _always_false_v = false;

enum class MemoryUnit { BYTE, KiB, MiB, GiB, TiB, KB, MB, GB, TB };

template <typename T, MemoryUnit units = MemoryUnit::MiB>
[[nodiscard]] constexpr double memory_size(std::size_t count) {
    const double size_in_bytes = count * sizeof(T); // cast to double is critical here
    if constexpr (units == MemoryUnit::BYTE) return size_in_bytes;
    else if constexpr (units == MemoryUnit::KiB) return size_in_bytes / 1024.;
    else if constexpr (units == MemoryUnit::MiB) return size_in_bytes / 1024. / 1024.;
    else if constexpr (units == MemoryUnit::GiB) return size_in_bytes / 1024. / 1024. / 1024.;
    else if constexpr (units == MemoryUnit::TiB) return size_in_bytes / 1024. / 1024. / 1024. / 1024.;
    else if constexpr (units == MemoryUnit::KB) return size_in_bytes / 1000.;
    else if constexpr (units == MemoryUnit::MB) return size_in_bytes / 1000. / 1000.;
    else if constexpr (units == MemoryUnit::GB) return size_in_bytes / 1000. / 1000. / 1000.;
    else if constexpr (units == MemoryUnit::TB) return size_in_bytes / 1000. / 1000. / 1000. / 1000.;
    else static_assert(_always_false_v<T>, "Function is a non-exhaustive visitor of enum class {MemoryUnit}.");
}

// ===============
// --- Meshing ---
// ===============

// Semantic helpers that allow user to directly pass both interval/point counts for grid subdivision,
// without thinking about whether function need +1 or -1 to its argument
struct Points {
    std::size_t count;

    Points() = delete;
    explicit Points(std::size_t count) : count(count) {}
};

struct Intervals {
    std::size_t count;

    Intervals() = delete;
    explicit Intervals(std::size_t count) : count(count) {}
    Intervals(Points points) : count(points.count - 1) {}
};

template <typename FloatType, std::enable_if_t<std::is_floating_point<FloatType>::value, bool> = true>
[[nodiscard]] std::vector<FloatType> linspace(FloatType L1, FloatType L2, Intervals N) {
    assert(L1 < L2);
    assert(N.count >= 1);

    const FloatType step = (L2 - L1) / N.count;

    std::vector<FloatType> res(N.count + 1);

    res[0] = L1;
    for (std::size_t i = 1; i < res.size(); ++i) res[i] = res[i - 1] + step;

    return res;
}

template <typename FloatType, typename FuncType,
          std::enable_if_t<std::is_floating_point<FloatType>::value, bool>                          = true,
          std::enable_if_t<is_function_with_signature<FuncType, FloatType(FloatType)>::value, bool> = true>
[[nodiscard]] FloatType integrate_trapezoidal(FuncType f, FloatType L1, FloatType L2, Intervals N) {
    assert(L1 < L2);
    assert(N.count >= 1);

    const FloatType step = (L2 - L1) / N.count;

    FloatType sum = 0;
    FloatType x   = L1;

    for (std::size_t i = 0; i < N.count; ++i, x += step) sum += f(x) + f(x + step);

    return FloatType(0.5) * sum * step;
}

// ====================
// --- Misc helpers ---
// ====================

template <typename UintType, std::enable_if_t<std::is_integral<UintType>::value, bool> = true>
[[nodiscard]] constexpr UintType uint_difference(UintType a, UintType b) {
    // Cast to widest type if there is a change values don't fit into a regular 'int'
    using WiderIntType = std::conditional_t<(sizeof(UintType) >= sizeof(int)), int64_t, int>;

    return static_cast<UintType>(utl::math::abs(static_cast<WiderIntType>(a) - static_cast<WiderIntType>(b)));
}

template <typename SizedContainer, std::enable_if_t<utl::math::is_sized<SizedContainer>::value, bool> = true>
[[nodiscard]] int ssize(const SizedContainer& container) {
    return static_cast<int>(container.size());
}

template <typename ArithmeticType, std::enable_if_t<std::is_arithmetic<ArithmeticType>::value, bool> = true>
[[nodiscard]] constexpr ArithmeticType ternary_branchless(bool condition, ArithmeticType return_if_true,
                                                          ArithmeticType return_if_false) {
    return (condition * return_if_true) + (!condition * return_if_false);
}

template <typename IntType, std::enable_if_t<std::is_integral<IntType>::value, bool> = true>
[[nodiscard]] constexpr IntType ternary_bitselect(bool condition, IntType return_if_true, IntType return_if_false) {
    return (return_if_true & -IntType(condition)) | (return_if_false & ~(-IntType(condition)));
}

template <typename IntType, std::enable_if_t<std::is_integral<IntType>::value, bool> = true>
[[nodiscard]] constexpr IntType ternary_bitselect(bool condition, IntType return_if_true) {
    return return_if_true & -IntType(condition);
}

} // namespace utl::math

#endif
#endif // module utl::math






// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DmitriBogdanov/prototyping_utils ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Module:        utl::mvl
// Documentation: https://github.com/DmitriBogdanov/prototyping_utils/blob/master/docs/module_mvl.md
// Source repo:   https://github.com/DmitriBogdanov/prototyping_utils
//
// This project is licensed under the MIT License
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#include <string_view>
#if !defined(UTL_PICK_MODULES) || defined(UTLMODULE_MVL)
#ifndef UTLHEADERGUARD_MVL
#define UTLHEADERGUARD_MVL

// _______________________ INCLUDES _______________________

#include <algorithm> // swap(), find(), count(), is_sorted(), min_element(),
                     // max_element(), sort(), stable_sort(), min(), max(), remove_if(), copy()
#include <cassert>   // assert() // Note: Perhaps temporary
#include <charconv>
#include <cmath>            // isfinite()
#include <cstddef>          // size_t, ptrdiff_t, nullptr_t
#include <functional>       // reference_wrapper<>, multiplies<>
#include <initializer_list> // initializer_list<>
#include <iomanip>          // setw()
#include <ios>              // right(), boolalpha(), ios::boolalpha
#include <iterator>         // random_access_iterator_tag, reverse_iterator<>
#include <memory>           // unique_ptr<>
#include <numeric>          // accumulate()
#include <ostream>          // ostream
#include <sstream>          // ostringstream
#include <stdexcept>        // out_of_range, invalid_argument
#include <string>           // string
#include <type_traits>      // conditional_t<>, enable_if_t<>, void_t<>, true_type, false_type, remove_reference_t<>
#include <utility>          // move()
#include <vector>           // vector<>

// ____________________ DEVELOPER DOCS ____________________

// This module tries to implement an "unreasonably flexible yet convenient" template for vectors and matrices,
// in order to do so we have to rely heavily on either conditional compilation or automatic code generation
// (or both). There are multiple seemingly viable approaches to it, yet the most of them proved to be either
// unsuitable or unreasonable from the implementation standpoint. Below is a little rundown of implementations
// that were attempted and used/discared for various reasons.
//
// Currently matrix/vector/view/etc code reuse is implemented though a whole buch of conditional compilation with
// SFINAE, abuse of conditional inheritance, conditional types and constexpr if's. While not perfect, this is
// the best working approach so far. Below is a list of approaches that has been considered & tried:
//
// 1) No code reuse, all classes implemented manually - unrealistically cumbersome, huge code duplication
//
//    => [-] UNSUITABLE APPROACH
//
// 2) Regular OOP with virtual classes - having vtables in lightweigh containers is highly undesirable, we can
//    avoid this using CRTP however we run into issues with multiple inheritance (see below)
//
//    => [-] UNSUITABLE APPROACH
//
// 3) CRTP - we unavoidably run into multiple inheritance issues (diamond problem) with concepts like:
//       [mutable-matrix-like] : [const-matrix-like] & [mutable-iterable] where both parents derive from
//       [const-iterable]
//
//    While those issues are resolvable, their handling becomes more and more difficult as parent classes grow larger.
//    A usual solution for this would be virtual inheritance, however it prevents us from calling derived methods inside
//    the base class due to "static downcast via virtual inheritance" rule, which makes 'static_cast<Derived*>(this)'
//    illegal. We can resolve all name collisions manually, however it leads to a code that is absolutely horrific and
//    easy to mess up.
//
//    There is also an issue of extreme boilerplate, since we want to propagate '_types<>' wrapper and the 'Final'
//    return type to all base classes we end up with extremely unwieldy template arguments that make compile errors and
//    refactoring a challenge.
//
//    => [~] SUITABLE, BUT VERY PROBLEMATIC APPROACH
//
// 4) Template-based mixins without CRTP - these allow to add functionality on top of a class, essentially it works
//    like "inverted" CRTP from the code logic POV. While being almost a perfect solution, it doesn't allow us to
//    return objects of derived type from base class methods, which is a requirement for the intended matrix API.
//
//    => [-] UNSUITABLE APPROACH
//
// 5) Macro-based mixins - insert functionality directly in a class, a simple way of doing exactly what we want.
//    "Mixed-in" methods are simply functions implemented in terms of certain "required" methods.
//    Doesn't lead to huge template chains and doesn't have any inheritance at all. The main downside is being
//    preprocessor based, which exposes some identifiers to a global namespace and makes all "connection points"
//    in logic dependant on correct naming inside macros, rather than actual semantical symbols.
//
//    Difficult to work with due to being ignored by the language server, gets increasingly cumbersome as the number of
//    classes grows and still has significant code duplication (vector/matrix * dense/strided/sparse *
//    * container/view/const_view = 18 manually created definitions).
//
//    => [+-] SUITABLE, BUT HEAVILY FLAWED APPROACH
//
// 6) A single class with all pars of API correctly enabled/disabled through SFINAE. Closes thing to a sensible
//    approach with minimal (basically none) code duplication. However it has a number of non-trivial quirks
//    when it comes to conditionally compiling member variables, types & functions (every group has a quirk of
//    its own, functions are the least problematic). Some of such quirks limit possible conditional API (member
//    types suffer from this) or, for example, prevent use of initialization lists on conditionally compiled
//    constructors (big sad).
//
//    A less obvious downside compared to macros, is that it doesn't get handles as well by the language server
//    autocomplete - SFINAE-disabled methods still show up as autocomplete suggestions, even if they get marked
//    as missing immediately upon writing the whole name.
//
//    => [+] DIFFICULT, BUT WORKABLE, BEST APPROACH SO FAR
//
// NOTES ON SPAN:
//
// We can easily implement Matlab-like API to take matrix blocks like this:
//    matrix(Span{0, 10}, 4)
// which would be equivalent to
//    matrix.block(0, 4, 10, 1)
// we only need a thin 'Span' POD with 2 members and a few overloads for 'operator()' that redirect span
// to 'this->block(...)'
//
// NOTES ON NAMING:
//
// Macro naming is a bit of a mess as of now.

// ____________________ IMPLEMENTATION ____________________

namespace utl::mvl {

// ===================
// --- Type Traits ---
// ===================

// MARK:
// Macro for generating type traits with all their boilerplate
//
// While it'd be nice to avoid macro usage alltogether, having a few macros for generating standardized boilerplate
// that gets repeated several dosen times DRASTICALLY improves the maintainability of the whole conditional compilation
// mechanism down the line. They will be later #undef'ed.

#define utl_mvl_define_trait(trait_name_, ...)                                                                         \
    template <class T, class = void>                                                                                   \
    struct trait_name_ : std::false_type {};                                                                           \
                                                                                                                       \
    template <class T>                                                                                                 \
    struct trait_name_<T, std::void_t<decltype(__VA_ARGS__)>> : std::true_type {};                                     \
                                                                                                                       \
    template <class T>                                                                                                 \
    constexpr bool trait_name_##_v = trait_name_<T>::value;                                                            \
                                                                                                                       \
    template <class T>                                                                                                 \
    using trait_name_##_enable_if = std::enable_if_t<trait_name_<T>::value, bool>

// Shortcuts for different types of requirements

#define utl_mvl_define_trait_has_binary_op(trait_name_, op_)                                                           \
    utl_mvl_define_trait(trait_name_, std::declval<std::decay_t<T>>() op_ std::declval<std::decay_t<T>>())

#define utl_mvl_define_trait_has_assignment_op(trait_name_, op_)                                                       \
    utl_mvl_define_trait(trait_name_, std::declval<std::decay_t<T>&>() op_ std::declval<std::decay_t<T>>())
// for operators like '+=' lhs should be a reference

#define utl_mvl_define_trait_has_unary_op(trait_name_, op_)                                                            \
    utl_mvl_define_trait(trait_name_, op_ std::declval<std::decay_t<T>>())

#define utl_mvl_define_trait_has_member(trait_name_, member_)                                                          \
    utl_mvl_define_trait(trait_name_, std::declval<std::decay_t<T>>().member_)

#define utl_mvl_define_trait_has_member_type(trait_name_, member_)                                                     \
    utl_mvl_define_trait(trait_name_, std::declval<typename std::decay_t<T>::member_>())

// --- Type traits ---
// -------------------

utl_mvl_define_trait(_has_ostream_output_op, std::declval<std::ostream>() << std::declval<T>());

utl_mvl_define_trait_has_binary_op(_has_binary_op_plus, +);
utl_mvl_define_trait_has_binary_op(_has_binary_op_minus, -);
utl_mvl_define_trait_has_binary_op(_has_binary_op_multiplies, *);
utl_mvl_define_trait_has_binary_op(_has_binary_op_less, <);
utl_mvl_define_trait_has_binary_op(_has_binary_op_greater, >);
utl_mvl_define_trait_has_binary_op(_has_binary_op_equal, ==);

utl_mvl_define_trait_has_assignment_op(_has_assignment_op_plus, +=);
utl_mvl_define_trait_has_assignment_op(_has_assignment_op_minus, -=);
utl_mvl_define_trait_has_assignment_op(_has_assignment_op_multiplies, *=);

utl_mvl_define_trait_has_unary_op(_has_unary_op_plus, +);
utl_mvl_define_trait_has_unary_op(_has_unary_op_minus, -);

utl_mvl_define_trait_has_member(_has_member_i, i);
utl_mvl_define_trait_has_member(_has_member_j, j);
utl_mvl_define_trait_has_member(_has_member_value, value);

utl_mvl_define_trait_has_member(_is_tensor, is_tensor);
utl_mvl_define_trait_has_member(_is_sparse_entry_1d, is_sparse_entry_1d);
utl_mvl_define_trait_has_member(_is_sparse_entry_2d, is_sparse_entry_2d);

// MARK:

// =======================
// --- Stringification ---
// =======================

// Stringification implementation from 'utl::log' module, see its source for more notes.

// --- Internal type traits ---
// ----------------------------

utl_mvl_define_trait(_has_begin, ++std::declval<T>().begin());
utl_mvl_define_trait(_has_end, ++std::declval<T>().end());
utl_mvl_define_trait(_has_get, std::get<0>(std::declval<T>()));
utl_mvl_define_trait(_has_tuple_size, std::tuple_size<T>::value);
utl_mvl_define_trait(_has_ostream_insert, std::declval<std::ostream>() << std::declval<T>());

// --- Internal utils ---
// ----------------------

template <class>
inline constexpr bool _always_false_v = false;

template <typename T>
constexpr int _log_10_ceil(T num) {
    return num < 10 ? 1 : 1 + _log_10_ceil(num / 10);
}

template <typename T>
constexpr std::size_t _max_float_digits =
    4 + std::numeric_limits<T>::max_digits10 + std::max(2, _log_10_ceil(std::numeric_limits<T>::max_exponent10));

// --- Stringifiers ---
// --------------------

template <class T>
void append_stringified(std::string& str, const T& value);

void _append_stringified_bool(std::string& str, bool value) { str += value ? "true" : "false"; }

template <class T>
void _append_stringified_integer(std::string& str, T value) {
    std::array<char, std::numeric_limits<T>::digits10> buffer;
    const auto [number_end_ptr, error_code] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);

    if (error_code != std::errc())
        throw std::runtime_error(
            "Integer stringification encountered std::to_chars() formatting error while serializing a value.");

    str.append(buffer.data(), number_end_ptr - buffer.data());
}

template <class T>
void _append_stringified_float(std::string& str, T value) {
    std::array<char, _max_float_digits<T>> buffer;
    constexpr int                          precision = 5;
    // in 'mvl' specifically we reduce default precision to make matrices format nices,
    // if user wants full precision they can always pass another stringifier, for example, `std::to_string()`
    const auto [number_end_ptr, error_code] =
        std::to_chars(buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::general, precision);

    if (error_code != std::errc())
        throw std::runtime_error(
            "Float stringification encountered std::to_chars() formatting error while serializing a value.");

    str.append(buffer.data(), number_end_ptr - buffer.data());
}

template <class T>
void _append_stringified_stringlike(std::string& str, const T& value) {
    str += value;
}

template <class T>
void _append_stringified_string_convertible(std::string& str, const T& value) {
    str += std::string(value);
}

template <class T>
void _append_stringified_array(std::string& str, const T& value) {
    str += "{ ";
    if (value.begin() != value.end())
        for (auto it = value.begin();;) {
            append_stringified(str, *it);
            if (++it == value.end()) break;
            str += ", ";
        }
    str += " }";
}

template <class Tuplelike, std::size_t... Idx>
void _append_stringified_tuple_impl(std::string& str, Tuplelike value, std::index_sequence<Idx...>) {
    ((Idx == 0 ? "" : str += ", ", append_stringified(str, std::get<Idx>(value))), ...);
}

template <template <class...> class Tuplelike, class... Args>
void _append_stringified_tuple(std::string& str, const Tuplelike<Args...>& value) {
    str += "< ";
    _append_stringified_tuple_impl(str, value, std::index_sequence_for<Args...>{});
    str += " >";
}

template <class T>
void _append_stringified_printable(std::string& str, const T& value) {
    str += (std::ostringstream() << value).str();
}

// --- Public API ---
// ------------------

template <class T>
void append_stringified(std::string& str, const T& value) {
    if constexpr (std::is_same_v<T, bool>) _append_stringified_bool(str, value);
    else if constexpr (std::is_same_v<T, char>) _append_stringified_stringlike(str, value);
    else if constexpr (std::is_integral_v<T>) _append_stringified_integer(str, value);
    else if constexpr (std::is_floating_point_v<T>) _append_stringified_float(str, value);
    else if constexpr (std::is_convertible_v<T, std::string_view>) _append_stringified_stringlike(str, value);
    else if constexpr (std::is_convertible_v<T, std::string>) _append_stringified_string_convertible(str, value);
    else if constexpr (_has_begin_v<T> && _has_end_v<T>) _append_stringified_array(str, value);
    else if constexpr (_has_get_v<T> && _has_tuple_size_v<T>) _append_stringified_tuple(str, value);
    else if constexpr (_has_ostream_insert_v<T>) _append_stringified_printable(str, value);
    else static_assert(_always_false_v<T>, "No valid stringification exists for the type.");
}

template <class... Args>
std::string stringify(Args&&... args) {
    std::string buffer;
    (append_stringified(buffer, std::forward<Args>(args)), ...);
    return buffer;
}

// We wrap stringifying function in functor-class so we can use it a default template callable argument.
// Templates can't infer template parameters from default arguments:
//
//    template<class Func>
//    void do_stuff(Func f = default_f);  // <- CAN'T infer 'Func'
//
//    template<class Func = default_functor>
//    void do_stuff(Func f = Func());     // <- CAN infer 'Func'
//
// which is why the "standard" way of doing it (standard as in used by STL containers) is to use functors as default
// template arguments, if user passes a callable it will override the 'Func' and we get the usual behaviour.
//
template <class T>
struct default_stringifier {
    [[nodiscard]] std::string operator()(const T& value) const { return stringify(value); }
};

// ======================
// --- Codegen Macros ---
// ======================

#define utl_mvl_assert(condition_) assert(condition_)
// if (!__VA_ARGS__) throw std::runtime_error("Failed assert on line " + std::to_string(__LINE__))

// ========================
// --- Helper Functions ---
// ========================

// Shortuct for labda-type-based SFINAE.
//
// Callables in this module are usually takes as a template type since 'std::function<>' introduces very significant
// overhead with its type erasure. With template args all lambdas and functors can be nicely inlined, however we lose
// the ability to overload functions that take callable the way we could with 'std::function<>'.
//
// As a workaround we can use SFINAE to reject "overloads" tha don't have a particular signature, effectively achieving
// the behaviour we need.
template <class FuncType, class Signature>
using _has_signature_enable_if = std::enable_if_t<std::is_convertible_v<FuncType, std::function<Signature>>, bool>;

template <class T>
[[nodiscard]] std::unique_ptr<T[]> _make_unique_ptr_array(size_t size) {
    return std::unique_ptr<T[]>(new T[size]);
}

// Marker for uncreachable code
[[noreturn]] inline void _unreachable() {
// (Implementation from https://en.cppreference.com/w/cpp/utility/unreachable)
// Use compiler specific extensions if possible.
// Even if no extension is used, undefined behavior is still raised by
// an empty function body and the noreturn attribute.
#if defined(_MSC_VER) && !defined(__clang__) // MSVC
    __assume(false);
#else // GCC, Clang
    __builtin_unreachable();
#endif
}

// =======================
// --- Utility Classes ---
// =======================

// Wrapper that allows passing standard set of member types down the CRTP chain
// (types have to be passed through template args one way or another. Trying to access
// 'derived_type::value_type' from the CRTP base class will fail, since in order for it to be available
// the derived class has to be "implemented" which can only happen after the base class is "implemented",
// which requires the type => a logical loop. So we wrap all the types in a class a pass them down
// the chain a "pack with everything" type of deal. This pack then gets "unwrapped" with
// '_utl_storage_define_types' macro in every class => we have all the member types defined).
//
// (!) Since CRTP approach has been deprecated in favor of mixins, the original reasoning is no longer
// accurate, however it is this a useful class whenever a pack of types have to be passed through a
// template (like, for example, in iterator implementation)
template <class T>
struct _types {
    using value_type      = T;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference       = T&;
    using const_reference = const T&;
    using pointer         = T*;
    using const_pointer   = const T*;
};

// A minimal equivalent of the once proposed 'std::observer_ptr<>'.
// Used in views to allow observer pointers with the same interface as 'std::unique_ptr',
// which means a generic '.data()` can be implemented without having to make a separate version for views and containers
template <class T>
class _observer_ptr {
public:
    using element_type = T;

private:
    element_type* _data = nullptr;

public:
    // - Constructors -
    constexpr _observer_ptr() noexcept = default;
    constexpr _observer_ptr(std::nullptr_t) noexcept {}
    explicit _observer_ptr(element_type* ptr) : _data(ptr) {}

    _observer_ptr& operator=(element_type* ptr) {
        this->_data = ptr;
        return *this;
    }

    // - Interface -
    [[nodiscard]] constexpr element_type* get() const noexcept { return _data; }
};

// =================
// --- Iterators ---
// =================

// Iterator template for 1D iterator over a tensor.
// Reduces code duplication for const & non-const variants by bundling some conditional logic.
// Uses 'operator[]' of 'ParentPointer' for all logic, thus allowing arbitrary tensors that
// don't have to be contiguous or even ordered in memory.
template <class ParentPointer, class Types, bool is_const_iter = false>
class _flat_iterator {
    using parent_pointer = ParentPointer;

public:
    // Iterator reqs: [General]
    // Contains member types      =>
    using iterator_category = std::random_access_iterator_tag;
    using difference_type   = typename Types::difference_type;
    using value_type        = typename Types::value_type;
    using pointer = typename std::conditional_t<is_const_iter, typename Types::const_pointer, typename Types::pointer>;
    using reference =
        typename std::conditional_t<is_const_iter, typename Types::const_reference, typename Types::reference>;

    // Iterator reqs: [General]
    // Constructible
    _flat_iterator(parent_pointer parent, difference_type idx) : _parent(parent), _idx(idx) {}
    // Copy-constructible => by default
    // Copy-assignable    => by default
    // Destructible       => by default
    // Swappable
    friend void swap(_flat_iterator& lhs, _flat_iterator& rhs) { std::swap(lhs._idx, rhs._idx); }

    // Iterator reqs: [General]
    // Can be incremented (prefix & postfix)
    _flat_iterator& operator++() {
        ++_idx;
        return *this;
    } // prefix
    _flat_iterator operator++(int) {
        _flat_iterator temp = *this;
        ++_idx;
        return temp;
    } // postfix

    // Iterator reqs: [Input iterator]
    // Supports equality/inequality comparisons
    friend bool operator==(const _flat_iterator& it1, const _flat_iterator& it2) { return it1._idx == it2._idx; };
    friend bool operator!=(const _flat_iterator& it1, const _flat_iterator& it2) { return it1._idx != it2._idx; };
    // Can be dereferenced as an rvalue
    reference   operator*() const { return _parent->operator[](_idx); }
    pointer     operator->() const { return &_parent->operator[](_idx); }

    // Iterator reqs: [Output iterator]
    // Can be dereferenced as an lvalue (only for mutable iterator types) => not needed

    // Iterator reqs: [Forward iterator]
    // Default-constructible
    _flat_iterator() : _parent(nullptr), _idx(0) {}
    // "Multi-pass" - dereferencing & incrementing does not affects dereferenceability => satisfied

    // Iterator reqs: [Bidirectional iterator]
    // Can be decremented
    _flat_iterator& operator--() {
        --_idx;
        return *this;
    } // prefix
    _flat_iterator operator--(int) {
        _flat_iterator temp = *this;
        --_idx;
        return temp;
    } // postfix

    // Iterator reqs: [Random Access iterator]
    // See: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
    // Supports arithmetic operators: it + n, n + it, it - n, it1 - it2
    friend _flat_iterator operator+(const _flat_iterator& it, difference_type diff) {
        return _flat_iterator(it._parent, it._idx + diff);
    }
    friend _flat_iterator operator+(difference_type diff, const _flat_iterator& it) {
        return _flat_iterator(it._parent, it._idx + diff);
    }
    friend _flat_iterator operator-(const _flat_iterator& it, difference_type diff) {
        return _flat_iterator(it._parent, it._idx - diff);
    }
    friend difference_type operator-(const _flat_iterator& it1, const _flat_iterator& it2) {
        return it1._idx - it2._idx;
    }
    // Supports inequality comparisons (<, >, <= and >=) between iterators
    friend bool     operator<(const _flat_iterator& it1, const _flat_iterator& it2) { return it1._idx < it2._idx; }
    friend bool     operator>(const _flat_iterator& it1, const _flat_iterator& it2) { return it1._idx > it2._idx; }
    friend bool     operator<=(const _flat_iterator& it1, const _flat_iterator& it2) { return it1._idx <= it2._idx; }
    friend bool     operator>=(const _flat_iterator& it1, const _flat_iterator& it2) { return it1._idx >= it2._idx; }
    // Standard assumption: Both iterators are from the same container => no need to compare '_parent'
    // Supports compound assignment operations += and -=
    _flat_iterator& operator+=(difference_type diff) {
        _idx += diff;
        return *this;
    }
    _flat_iterator& operator-=(difference_type diff) {
        _idx -= diff;
        return *this;
    }
    // Supports offset dereference operator ([])
    reference operator[](difference_type diff) const { return _parent->operator[](_idx + diff); }

private:
    parent_pointer  _parent;
    difference_type _idx; // not size_type because we have to use it in 'difference_type' operations most of the time
};

// ======================================
// --- Sparse Matrix Pairs & Triplets ---
// ======================================

// Note:
// All sparse entries and multi-dimensional indeces can be sorted lexicographically

template <typename T>
struct SparseEntry1D {
    size_t i;
    T      value;

    constexpr static bool is_sparse_entry_1d = true;

    [[nodiscard]] bool operator<(const SparseEntry1D& other) const noexcept { return this->i < other.i; }
    [[nodiscard]] bool operator>(const SparseEntry1D& other) const noexcept { return this->i > other.i; }
};

template <typename T>
struct SparseEntry2D {
    size_t i;
    size_t j;
    T      value;

    constexpr static bool is_sparse_entry_2d = true;

    [[nodiscard]] bool operator<(const SparseEntry2D& other) const noexcept {
        return (this->i < other.i) || (this->j < other.j);
    }
    [[nodiscard]] bool operator>(const SparseEntry2D& other) const noexcept {
        return (this->i > other.i) || (this->j > other.j);
    }
};

struct Index2D {
    size_t i;
    size_t j;

    [[nodiscard]] bool operator<(const Index2D& other) const noexcept {
        return (this->i < other.i) || (this->j < other.j);
    }
    [[nodiscard]] bool operator>(const Index2D& other) const noexcept {
        return (this->i > other.i) || (this->j > other.j);
    }
    [[nodiscard]] bool operator==(const Index2D& other) const noexcept {
        return (this->i == other.i) && (this->j == other.j);
    }
};

// Functions that take 2 sparse entries '{ i, j, v1 }' and '{ i, j, v2 }' and return '{ i, j, op(v1, v2) }'.
// Same thing for 1D sparse entries. LHS/RHS index correctness only gets checked in debug.
//
// Declaring these functions saves us from having to duplicate implementations of 'apply_binary_operator()'
// for 1D and 2D sparse entries that have different number of constructor args. Instead, those cases will
// be handled by '_apply_binary_op_to_sparse_entry()' that is SFINAE-overloaded to work with both.
//
// Implementations use perfect forwarding for everything and do the "overloading" through SFINAE with type traits.
//
// In reality 'op' doesn't ever get forwarded as an r-value but since are doing things generic why not forward it
// properly as well. This is also somewhat ugly, but it works well enough and there isn't much reason to rewrite
// this with some new abstraction.
//
template <class L, class R, class Op, _is_sparse_entry_1d_enable_if<L> = true, _is_sparse_entry_1d_enable_if<R> = true>
std::decay_t<L> _apply_binary_op_to_sparse_entries(L&& left, R&& right, Op&& op) {
    utl_mvl_assert(left.i == right.i);
    return {left.i, std::forward<Op>(op)(std::forward<L>(left).value, std::forward<R>(right).value)};
}

template <class L, class R, class Op, _is_sparse_entry_2d_enable_if<L> = true, _is_sparse_entry_2d_enable_if<R> = true>
std::decay_t<L> _apply_binary_op_to_sparse_entries(L&& left, R&& right, Op&& op) {
    utl_mvl_assert(left.i == right.i);
    utl_mvl_assert(left.j == right.j);
    return {left.i, left.j, std::forward<Op>(op)(std::forward<L>(left).value, std::forward<R>(right).value)};
}

template <class L, class R, class Op, _is_sparse_entry_1d_enable_if<L> = true>
std::decay_t<L> _apply_binary_op_to_sparse_entry_and_value(L&& left, R&& right_value, Op&& op) {
    return {left.i, std::forward<Op>(op)(std::forward<L>(left).value, std::forward<R>(right_value))};
}

template <class L, class R, class Op, _is_sparse_entry_2d_enable_if<L> = true>
std::decay_t<L> _apply_binary_op_to_sparse_entry_and_value(L&& left, R&& right_value, Op&& op) {
    return {left.i, left.j, std::forward<Op>(op)(std::forward<L>(left).value, std::forward<R>(right_value))};
}

template <class L, class R, class Op, _is_sparse_entry_1d_enable_if<R> = true>
std::decay_t<R> _apply_binary_op_to_value_and_sparse_entry(L&& left_value, R&& right, Op&& op) {
    return {right.i, std::forward<Op>(op)(std::forward<L>(left_value), std::forward<R>(right).value)};
}

template <class L, class R, class Op, _is_sparse_entry_2d_enable_if<R> = true>
std::decay_t<R> _apply_binary_op_to_value_and_sparse_entry(L&& left_value, R&& right, Op&& op) {
    return {right.i, right.j, std::forward<Op>(op)(std::forward<L>(left_value), std::forward<R>(right).value)};
}
// MARK:

// =============
// --- Enums ---
// =============

// Parameter enums
//
// Their combination specifies compiled tensor API.
enum class Dimension { VECTOR, MATRIX };
enum class Type { DENSE, STRIDED, SPARSE };
enum class Ownership { CONTAINER, VIEW, CONST_VIEW };

// Config enums
//
// They specify conditional logic for a tensor.
// Combination of config & parameter enums fully defines a GenericTensor.
enum class Checking { NONE, BOUNDS };
enum class Layout { /* 1D */ FLAT, /* 2D */ RC, CR, /* Other */ SPARSE };

// Shortcut template used to deduce type of '_data' based on tensor 'ownership' parameter
template <Ownership ownership, class ContainerResult, class ViewResult, class ConstViewResult>
using _choose_based_on_ownership =
    std::conditional_t<ownership == Ownership::CONTAINER, ContainerResult,
                       std::conditional_t<ownership == Ownership::VIEW, ViewResult, ConstViewResult>>;

// Shortcut template used to check that both arguments are tensors
// that have the same value type, mostly used in binary operators
// for nicer error messages and to prevent accidental conversions
template <class L, class R>
using _are_tensors_with_same_value_type_enable_if =
    std::enable_if_t<_is_tensor_v<L> && _is_tensor_v<R> &&
                         std::is_same_v<typename std::decay_t<L>::value_type, typename std::decay_t<R>::value_type>,
                     bool>;

// =====================================
// --- Boilerplate generation macros ---
// =====================================

// Macros used to pass around unwieldy chains of tensor template arguments.
//
#define utl_mvl_tensor_arg_defs                                                                                        \
    class T, Dimension _dimension, Type _type, Ownership _ownership, Checking _checking, Layout _layout

#define utl_mvl_tensor_arg_vals T, _dimension, _type, _ownership, _checking, _layout

// Incredibly improtant macros used for conditional compilation of member functions.
// They automatically create the boilerplate that makes member functions dependant on the template parameters,
// which is necessary for conditional compilation and "forward" to a 'enable_if' condition.
//
// 'utl_mvl_require' is intended to be used inside template methods (template in a sense that they have other template
// args besides this conditional compilation boilerplate which technically makes all methods template).
//
// 'utl_mvl_reqs' is used when we want only conditional compilation (aka majority of cases) and cuts down on
// boilerplate even more.
//
#define utl_mvl_require(condition_)                                                                                    \
    typename value_type = T, Dimension dimension = _dimension, Type type = _type, Ownership ownership = _ownership,    \
             Checking checking = _checking, Layout layout = _layout, std::enable_if_t<condition_, bool> = true

#define utl_mvl_reqs(condition_) template <utl_mvl_require(condition_)>

// A somewhat scuffed version of trait-definig macro used to create SFINAE-restrictions
// on tensor params in free functions. Only supports trivial conditions of the form
// '<parameter> [==][!=] <value>'. Perhaps there is a better way of doing it, but I'm not yet sure.
//
// Used mostly to restrict linear algebra operations for sparse/nonsparse so we can get
// a "SFINAE-driven overloading" on operators that take both arguments with perfect forwarding,
// which means they can't be overloaded in a regular way
//
#define utl_mvl_define_tensor_param_restriction(trait_name_, expr_)                                                    \
    template <class T>                                                                                                 \
    constexpr bool trait_name_##_v = (std::decay_t<T>::params::expr_);                                                 \
                                                                                                                       \
    template <class T>                                                                                                 \
    using trait_name_##_enable_if = std::enable_if_t<trait_name_##_v<T>, bool>;

utl_mvl_define_tensor_param_restriction(_is_sparse_tensor, type == Type::SPARSE);
utl_mvl_define_tensor_param_restriction(_is_nonsparse_tensor, type != Type::SPARSE);
utl_mvl_define_tensor_param_restriction(_is_matrix_tensor, dimension == Dimension::MATRIX);

// ===========================
// --- Data Member Classes ---
// ===========================

// Unlike class method, member values can't be templated, which prevents us from using regular 'enable_if_t' SFINAE
// for their conditional compilation. The (seemingly) best workaround to compile members conditionally is to inherit
// 'std::contidional<T, EmptyClass>' where 'T' is a "dummy" class with the sole purpose of having data members to
// inherit. This does not introduce virtualiztion (which is good, that how we want it).

template <int id>
struct _nothing {};

template <utl_mvl_tensor_arg_defs>
class _2d_extents {
private:
    using size_type = typename _types<T>::size_type;

public:
    size_type _rows = 0;
    size_type _cols = 0;
};

template <utl_mvl_tensor_arg_defs>
class _2d_strides {
private:
    using size_type = typename _types<T>::size_type;

public:
    size_type _row_stride = 0;
    size_type _col_stride = 0;
};

template <utl_mvl_tensor_arg_defs>
struct _2d_dense_data {
private:
    using value_type = typename _types<T>::value_type;
    using _data_t    = _choose_based_on_ownership<_ownership, std::unique_ptr<value_type[]>, _observer_ptr<value_type>,
                                               _observer_ptr<const value_type>>;

public:
    _data_t _data;
};

template <utl_mvl_tensor_arg_defs>
struct _2d_sparse_data {
private:
    using value_type = typename _types<T>::value_type;
    using _triplet_t = _choose_based_on_ownership<_ownership, SparseEntry2D<value_type>,
                                                  SparseEntry2D<std::reference_wrapper<value_type>>,
                                                  SparseEntry2D<std::reference_wrapper<const value_type>>>;

public:
    using triplet_type = _triplet_t;

    std::vector<triplet_type> _data;
};

// ===================
// --- Tensor Type ---
// ===================

template <utl_mvl_tensor_arg_defs>
class GenericTensor
    // Conditionally compile member variables through inheritance
    : public std::conditional_t<_dimension == Dimension::MATRIX, _2d_extents<utl_mvl_tensor_arg_vals>, _nothing<1>>,
      public std::conditional_t<_dimension == Dimension::MATRIX && _type == Type::STRIDED,
                                _2d_strides<utl_mvl_tensor_arg_vals>, _nothing<2>>,
      public std::conditional_t<_type == Type::DENSE || _type == Type::STRIDED, _2d_dense_data<utl_mvl_tensor_arg_vals>,
                                _nothing<3>>,
      public std::conditional_t<_type == Type::SPARSE, _2d_sparse_data<utl_mvl_tensor_arg_vals>, _nothing<4>>
// > After this point no non-static member variables will be introduced
{
    // --- Parameter reflection ---
    // ----------------------------
public:
    struct params {
        constexpr static auto dimension = _dimension;
        constexpr static auto type      = _type;
        constexpr static auto ownership = _ownership;
        constexpr static auto checking  = _checking;
        constexpr static auto layout    = _layout;

        // Prevent impossible layouts
        static_assert((dimension == Dimension::VECTOR) == (layout == Layout::FLAT), "Flat layout <=> matrix is 1D.");
        static_assert((type == Type::SPARSE) == (layout == Layout::SPARSE), "Sparse layout <=> matrix is sparse.");
    };

    constexpr static bool is_tensor = true;

    // --- Member types ---
    // --------------------
private:
    using _type_wrapper = _types<T>;

public:
    using self            = GenericTensor;
    using value_type      = typename _type_wrapper::value_type;
    using size_type       = typename _type_wrapper::size_type;
    using difference_type = typename _type_wrapper::difference_type;
    using reference       = typename _type_wrapper::reference;
    using const_reference = typename _type_wrapper::const_reference;
    using pointer         = typename _type_wrapper::pointer;
    using const_pointer   = typename _type_wrapper::const_pointer;

    using owning_reflection = GenericTensor<value_type, params::dimension, params::type, Ownership::CONTAINER,
                                            params::checking, params::layout>;
    // container type corresponding to 'self', this is the return type of algebraic operations on a tensor

    // --- Iterators ---
    // -----------------
public:
    using const_iterator         = _flat_iterator<const self*, _type_wrapper, true>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    using iterator = std::conditional_t<_ownership != Ownership::CONST_VIEW, _flat_iterator<self*, _type_wrapper>,
                                        const_iterator>; // for const views 'iterator' is the same as 'const_iterator'
    using reverse_iterator = std::reverse_iterator<iterator>;

    [[nodiscard]] const_iterator cbegin() const { return const_iterator(this, 0); }
    [[nodiscard]] const_iterator cend() const { return const_iterator(this, this->size()); }
    [[nodiscard]] const_iterator begin() const { return this->cbegin(); }
    [[nodiscard]] const_iterator end() const { return this->cend(); }

    [[nodiscard]] const_reverse_iterator crbegin() const { return const_reverse_iterator(this->cend()); }
    [[nodiscard]] const_reverse_iterator crend() const { return const_reverse_iterator(this->cbegin()); }
    [[nodiscard]] const_reverse_iterator rbegin() const { return this->crbegin(); }
    [[nodiscard]] const_reverse_iterator rend() const { return this->crend(); }

    utl_mvl_reqs(ownership != Ownership::CONST_VIEW) [[nodiscard]] iterator begin() { return iterator(this, 0); }

    utl_mvl_reqs(ownership != Ownership::CONST_VIEW) [[nodiscard]] iterator end() {
        return iterator(this, this->size());
    }

    utl_mvl_reqs(ownership != Ownership::CONST_VIEW) [[nodiscard]] reverse_iterator rbegin() {
        return reverse_iterator(this->end());
    }

    utl_mvl_reqs(ownership != Ownership::CONST_VIEW) [[nodiscard]] reverse_iterator rend() {
        return reverse_iterator(this->begin());
    }

    // --- Basic getters ---
    // ---------------------
public:
    utl_mvl_reqs(dimension == Dimension::MATRIX && (type == Type::DENSE || type == Type::STRIDED))
        [[nodiscard]] size_type size() const noexcept {
        return this->rows() * this->cols();
    }

    utl_mvl_reqs(type == Type::SPARSE) [[nodiscard]] size_type size() const noexcept { return this->_data.size(); }

    utl_mvl_reqs(dimension == Dimension::MATRIX) [[nodiscard]] size_type rows() const noexcept { return this->_rows; }

    utl_mvl_reqs(dimension == Dimension::MATRIX) [[nodiscard]] size_type cols() const noexcept { return this->_cols; }

    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::DENSE) [[nodiscard]] constexpr size_type
        row_stride() const noexcept {
        if constexpr (self::params::layout == Layout::RC) return 0;
        if constexpr (self::params::layout == Layout::CR) return 1;
        _unreachable();
    }

    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::DENSE) [[nodiscard]] constexpr size_type
        col_stride() const noexcept {
        if constexpr (self::params::layout == Layout::RC) return 1;
        if constexpr (self::params::layout == Layout::CR) return 0;
        _unreachable();
    }
    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::STRIDED) [[nodiscard]] size_type
        row_stride() const noexcept {
        return this->_row_stride;
    }

    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::STRIDED) [[nodiscard]] size_type
        col_stride() const noexcept {
        return this->_col_stride;
    }

    utl_mvl_reqs(type == Type::DENSE || type == Type::STRIDED) [[nodiscard]] const_pointer data() const noexcept {
        return this->_data.get();
    }

    utl_mvl_reqs(ownership != Ownership::CONST_VIEW && (type == Type::DENSE || type == Type::STRIDED))
        [[nodiscard]] pointer data() noexcept {
        return this->_data.get();
    }

    [[nodiscard]] bool empty() const noexcept { return (this->size() == 0); }

    // --- Advanced getters ---
    // ------------------------
    utl_mvl_reqs(_has_binary_op_equal<value_type>::value) [[nodiscard]] bool contains(const_reference value) const {
        return std::find(this->cbegin(), this->cend(), value) != this->cend();
    }

    utl_mvl_reqs(_has_binary_op_equal<value_type>::value) [[nodiscard]] size_type count(const_reference value) const {
        return std::count(this->cbegin(), this->cend(), value);
    }

    utl_mvl_reqs(_has_binary_op_less<value_type>::value) [[nodiscard]] bool is_sorted() const {
        return std::is_sorted(this->cbegin(), this->cend());
    }

    template <typename Compare>
    [[nodiscard]] bool is_sorted(Compare cmp) const {
        return std::is_sorted(this->cbegin(), this->cend(), cmp);
    }

    [[nodiscard]] std::vector<value_type> to_std_vector() const { return std::vector(this->cbegin(), this->cend()); }

    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::DENSE) self transposed() const {
        self res(this->cols(), this->rows());
        this->for_each([&](const value_type& element, size_type i, size_type j) { res(j, i) = element; });
        return res;
    }

    utl_mvl_reqs(ownership == Ownership::CONTAINER) [[nodiscard]] self clone() const { return *this; }

    utl_mvl_reqs(ownership == Ownership::CONTAINER) [[nodiscard]] self move() & { return std::move(*this); }

    template <Type other_type, Ownership other_ownership, Checking other_checking, Layout other_layout>
    [[nodiscard]] bool
    compare_contents(const GenericTensor<value_type, self::params::dimension, other_type, other_ownership,
                                         other_checking, other_layout>& other) const {
        // Surface-level checks
        if ((this->rows() != other.rows()) || (this->cols() != other.cols())) return false;
        // Compare while respecting sparsity
        constexpr bool is_sparse_l = (self::params::type == Type::SPARSE);
        constexpr bool is_sparse_r = (std::remove_reference_t<decltype(other)>::params::type == Type::SPARSE);
        // Same sparsity comparison
        if constexpr (is_sparse_l == is_sparse_r) {
            return this->size() == other.size() &&
                   this->true_for_all([&](const_reference e, size_type i, size_type j) { return e == other(i, j); });
        }
        // Different sparsity comparison
        // TODO: Impl here and use .all_of() OR .any_of()
        return true;
    }

    // --- Indexation ---
    // ------------------
public:
    // Vector API
    [[nodiscard]] const_reference front() const { return this->operator[](0); }
    [[nodiscard]] const_reference back() const { return this->operator[](this->size() - 1); }
    [[nodiscard]] reference       front() { return this->operator[](0); }
    [[nodiscard]] reference       back() { return this->operator[](this->size() - 1); }

private:
    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::STRIDED) [[nodiscard]] size_type
        _get_memory_offset_strided_impl(size_type idx, size_type i, size_type j) const {
        if constexpr (self::params::layout == Layout::RC) return idx * this->col_stride() + this->row_stride() * i;
        if constexpr (self::params::layout == Layout::CR) return idx * this->row_stride() + this->col_stride() * j;
        _unreachable();
    }

public:
    utl_mvl_reqs(type == Type::DENSE) [[nodiscard]] size_type get_memory_offset_of_idx(size_type idx) const {
        if constexpr (self::params::checking == Checking::BOUNDS) this->_bound_check_idx(idx);
        return idx;
    }

    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::DENSE) [[nodiscard]] size_type
        get_memory_offset_of_ij(size_type i, size_type j) const {
        return this->get_idx_of_ij(i, j);
    }

    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::STRIDED) [[nodiscard]] size_type
        get_memory_offset_of_idx(size_type idx) const {
        const auto ij = this->get_ij_of_idx(idx);
        return _get_memory_offset_strided_impl(idx, ij.i, ij.j);
    }

    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::STRIDED) [[nodiscard]] size_type
        get_memory_offset_of_ij(size_type i, size_type j) const {
        const auto idx = this->get_idx_of_ij(i, j);
        return _get_memory_offset_strided_impl(idx, i, j);
    }

public:
    // - Flat indexation -
    utl_mvl_reqs(ownership != Ownership::CONST_VIEW && dimension == Dimension::MATRIX &&
                 (type == Type::DENSE || type == Type::STRIDED)) [[nodiscard]] reference
    operator[](size_type idx) {
        return this->data()[this->get_memory_offset_of_idx(idx)];
    }

    utl_mvl_reqs(dimension == Dimension::MATRIX && (type == Type::DENSE || type == Type::STRIDED))
        [[nodiscard]] const_reference
        operator[](size_type idx) const {
        return this->data()[this->get_memory_offset_of_idx(idx)];
    }

    utl_mvl_reqs(ownership != Ownership::CONST_VIEW && dimension == Dimension::VECTOR || type == Type::SPARSE)
        [[nodiscard]] reference
        operator[](size_type idx) {
        if constexpr (self::params::checking == Checking::BOUNDS) this->_bound_check_idx(idx);
        return this->_data[idx].value;
    }

    utl_mvl_reqs(dimension == Dimension::VECTOR || type == Type::SPARSE) [[nodiscard]] const_reference
    operator[](size_type idx) const {
        if constexpr (self::params::checking == Checking::BOUNDS) this->_bound_check_idx(idx);
        return this->_data[idx].value;
    }

    // - 2D indexation -
    utl_mvl_reqs(ownership != Ownership::CONST_VIEW && dimension == Dimension::MATRIX &&
                 (type == Type::DENSE || type == Type::STRIDED)) [[nodiscard]] reference
    operator()(size_type i, size_type j) {
        return this->data()[this->get_memory_offset_of_ij(i, j)];
    }

    utl_mvl_reqs(dimension == Dimension::MATRIX && (type == Type::DENSE || type == Type::STRIDED))
        [[nodiscard]] const_reference
        operator()(size_type i, size_type j) const {
        return this->data()[this->get_memory_offset_of_ij(i, j)];
    }

    utl_mvl_reqs(ownership != Ownership::CONST_VIEW && dimension == Dimension::MATRIX && type == Type::SPARSE)
        [[nodiscard]] reference
        operator()(size_type i, size_type j) {
        if constexpr (self::params::checking == Checking::BOUNDS) this->_bound_check_idx(i, j);
        return this->_data[this->get_idx_of_ij(i, j)].value;
    }
    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::SPARSE) [[nodiscard]] const_reference
    operator()(size_type i, size_type j) const {
        if constexpr (self::params::checking == Checking::BOUNDS) this->_bound_check_idx(i, j);
        return this->_data[this->get_idx_of_ij(i, j)].value;
    }

    // --- Index conversions ---
    // -------------------------

    // - Bound checking -
private:
    void _bound_check_idx(size_type idx) const {
        if (idx >= this->size())
            throw std::out_of_range(
                stringify("idx (which is ", idx, ") >= this->size() (which is ", this->size(), ")"));
    }

    utl_mvl_reqs(dimension == Dimension::MATRIX) void _bound_check_ij(size_type i, size_type j) const {
        if (i >= this->rows())
            throw std::out_of_range(stringify("i (which is ", i, ") >= this->rows() (which is ", this->rows(), ")"));
        else if (j >= this->cols())
            throw std::out_of_range(stringify("j (which is ", j, ") >= this->cols() (which is ", this->cols(), ")"));
    }

    // - Dense & strided implementations -
private:
    utl_mvl_reqs(dimension == Dimension::MATRIX && (type == Type::DENSE || type == Type::STRIDED))
        [[nodiscard]] size_type _unchecked_get_idx_of_ij(size_type i, size_type j) const {
        if constexpr (self::params::layout == Layout::RC) return i * this->cols() + j;
        if constexpr (self::params::layout == Layout::CR) return j * this->rows() + i;
        _unreachable();
    }

    utl_mvl_reqs(dimension == Dimension::MATRIX && (type == Type::DENSE || type == Type::STRIDED)) [[nodiscard]] Index2D
        _unchecked_get_ij_of_idx(size_type idx) const {
        if constexpr (self::params::layout == Layout::RC) return {idx / this->cols(), idx % this->cols()};
        if constexpr (self::params::layout == Layout::CR) return {idx % this->rows(), idx / this->rows()};
        _unreachable();
    }

    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::STRIDED && ownership == Ownership::CONTAINER)
        [[nodiscard]] size_type _total_allocated_size() const noexcept {
        // Note 1: Allocated size of the strided matrix is NOT equal to .size() (which is same as rows * cols)
        // This is due to all the padding between the actual elements
        // Note 2: The question of whether .size() should return the number of 'strided' elements or the number
        // of actually allocated elements is rather perplexing, while the second option is more "correct" its
        // usefulness is dubious at bets, while the first one does in fact allow convenient 1D iteration over
        // all the elements. Option 1 also provides an API consistent with strided views, which is why it ended
        // up being the one chosen
        //
        // This method returns a true allocated size
        if constexpr (self::params::layout == Layout::RC)
            return (this->rows() - 1) * this->row_stride() + this->rows() * this->cols() * this->col_stride();
        if constexpr (self::params::layout == Layout::CR)
            return (this->cols() - 1) * this->col_stride() + this->rows() * this->cols() * this->row_stride();
        _unreachable();
    }

public:
    utl_mvl_reqs(dimension == Dimension::MATRIX && (type == Type::DENSE || type == Type::STRIDED))
        [[nodiscard]] size_type get_idx_of_ij(size_type i, size_type j) const {
        if constexpr (self::params::checking == Checking::BOUNDS) this->_bound_check_ij(i, j);
        return _unchecked_get_idx_of_ij(i, j);
    }

    utl_mvl_reqs(dimension == Dimension::MATRIX && (type == Type::DENSE || type == Type::STRIDED)) [[nodiscard]] Index2D
        get_ij_of_idx(size_type idx) const {
        if constexpr (self::params::checking == Checking::BOUNDS) this->_bound_check_idx(idx);
        return _unchecked_get_ij_of_idx(idx);
    }

    utl_mvl_reqs(dimension == Dimension::MATRIX && (type == Type::DENSE || type == Type::STRIDED))
        [[nodiscard]] size_type extent_major() const noexcept {
        if constexpr (self::params::layout == Layout::RC) return this->rows();
        if constexpr (self::params::layout == Layout::CR) return this->cols();
        _unreachable();
    }

    utl_mvl_reqs(dimension == Dimension::MATRIX && (type == Type::DENSE || type == Type::STRIDED))
        [[nodiscard]] size_type extent_minor() const noexcept {
        if constexpr (self::params::layout == Layout::RC) return this->cols();
        if constexpr (self::params::layout == Layout::CR) return this->rows();
        _unreachable();
    }

    // - Sparse implementations -
private:
    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::SPARSE) [[nodiscard]] size_type
        _search_ij(size_type i, size_type j) const noexcept {
        // Returns this->size() if {i, j} wasn't found.
        // Linear search for small .size() (more efficient fue to prediction and cache locality)
        if (true) {
            for (size_type idx = 0; idx < this->size(); ++idx)
                if (this->_data[idx].i == i && this->_data[idx].j == j) return idx;
            return this->size();
        }
        // TODO: Binary search for larger .size() (N(log2(size)) instead of N(size) asymptotically)
    }

public:
    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::SPARSE) [[nodiscard]] size_type
        get_idx_of_ij(size_type i, size_type j) const {
        const size_type idx = this->_search_ij(i, j);
        // Return this->size() if {i, j} wasn't found. Throw with bound checking.
        if constexpr (self::params::checking == Checking::BOUNDS)
            if (idx == this->size())
                throw std::out_of_range(stringify("Index { ", i, ", ", j, "} in not a part of sparse matrix"));
        return idx;
    }

    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::SPARSE) [[nodiscard]] Index2D
        get_ij_of_idx(size_type idx) const {
        if constexpr (self::params::checking == Checking::BOUNDS) this->_bound_check_idx(idx);
        return Index2D{this->_data[idx].i, this->_data[idx].j};
    }

    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::SPARSE)
        [[nodiscard]] bool contains_index(size_type i, size_type j) const noexcept {
        return this->_search_ij(i, j) != this->size();
    }

    // --- Reductions ---
    // ------------------
    utl_mvl_reqs(_has_binary_op_plus<value_type>::value) [[nodiscard]] value_type sum() const {
        return std::accumulate(this->cbegin(), this->cend(), value_type());
    }

    utl_mvl_reqs(_has_binary_op_multiplies<value_type>::value) [[nodiscard]] value_type product() const {
        return std::accumulate(this->cbegin(), this->cend(), value_type(), std::multiplies<value_type>());
    }

    utl_mvl_reqs(_has_binary_op_less<value_type>::value) [[nodiscard]] value_type min() const {
        return *std::min_element(this->cbegin(), this->cend());
    }

    utl_mvl_reqs(_has_binary_op_less<value_type>::value) [[nodiscard]] value_type max() const {
        return *std::max_element(this->cbegin(), this->cend());
    }

    // --- Predicate operations ---
    // ----------------------------
    template <class PredType, _has_signature_enable_if<PredType, bool(const_reference)> = true>
    [[nodiscard]] bool true_for_any(PredType predicate) const {
        for (size_type idx = 0; idx < this->size(); ++idx)
            if (predicate(this->operator[](idx), idx)) return true;
        return false;
    }

    template <class PredType, _has_signature_enable_if<PredType, bool(const_reference, size_type)> = true>
    [[nodiscard]] bool true_for_any(PredType predicate) const {
        for (size_type idx = 0; idx < this->size(); ++idx)
            if (predicate(this->operator[](idx), idx)) return true;
        return false;
    }

    template <class PredType, _has_signature_enable_if<PredType, bool(const_reference, size_type, size_type)> = true,
              utl_mvl_require(dimension == Dimension::MATRIX)>
    [[nodiscard]] bool true_for_any(PredType predicate) const {
        // Loop over all 2D indices using 1D loop with idx->ij conversion
        // This is just as fast and ensures looping only over existing elements in non-dense matrices
        for (size_type idx = 0; idx < this->size(); ++idx) {
            const auto ij = this->get_ij_of_idx(idx);
            if (predicate(this->operator[](idx), ij.i, ij.j)) return true;
        }
        return false;
    }

    template <class PredType, _has_signature_enable_if<PredType, bool(const_reference)> = true>
    [[nodiscard]] bool true_for_all(PredType predicate) const {
        auto inversed_predicate = [&](const_reference e) -> bool { return !predicate(e); };
        return !this->true_for_any(inversed_predicate);
    }

    template <class PredType, _has_signature_enable_if<PredType, bool(const_reference, size_type)> = true>
    [[nodiscard]] bool true_for_all(PredType predicate) const {
        auto inversed_predicate = [&](const_reference e, size_type idx) -> bool { return !predicate(e, idx); };
        return !this->true_for_any(inversed_predicate);
    }

    template <class PredType, _has_signature_enable_if<PredType, bool(const_reference, size_type, size_type)> = true,
              utl_mvl_require(dimension == Dimension::MATRIX)>
    [[nodiscard]] bool true_for_all(PredType predicate) const {
        // We can reuse .true_for_any() with inverted predicate due to following conjecture:
        // FOR_ALL (predicate)  ~  ! FOR_ANY (!predicate)
        auto inversed_predicate = [&](const_reference e, size_type i, size_type j) -> bool {
            return !predicate(e, i, j);
        };
        return !this->true_for_any(inversed_predicate);
    }

    // --- Const algorithms ---
    // ------------------------
    template <class FuncType, _has_signature_enable_if<FuncType, void(const_reference)> = true>
    const self& for_each(FuncType func) const {
        for (size_type idx = 0; idx < this->size(); ++idx) func(this->operator[](idx));
        return *this;
    }

    template <class FuncType, _has_signature_enable_if<FuncType, void(const_reference, size_type)> = true>
    const self& for_each(FuncType func) const {
        for (size_type idx = 0; idx < this->size(); ++idx) func(this->operator[](idx), idx);
        return *this;
    }

    template <class FuncType, _has_signature_enable_if<FuncType, void(const_reference, size_type, size_type)> = true,
              utl_mvl_require(dimension == Dimension::MATRIX)>
    const self& for_each(FuncType func) const {
        // Loop over all 2D indices using 1D loop with idx->ij conversion.
        // This is just as fast and ensures looping only over existing elements in non-dense matrices.
        for (size_type idx = 0; idx < this->size(); ++idx) {
            const auto ij = this->get_ij_of_idx(idx);
            func(this->operator[](idx), ij.i, ij.j);
        }
        return *this;
    }

    // --- Mutating algorithms ---
    // ---------------------------
    template <class FuncType, _has_signature_enable_if<FuncType, void(reference)> = true,
              utl_mvl_require(ownership != Ownership::CONST_VIEW)>
    self& for_each(FuncType func) {
        for (size_type idx = 0; idx < this->size(); ++idx) func(this->operator[](idx));
        return *this;
    }

    template <class FuncType, _has_signature_enable_if<FuncType, void(reference, size_type)> = true,
              utl_mvl_require(ownership != Ownership::CONST_VIEW)>
    self& for_each(FuncType func) {
        for (size_type idx = 0; idx < this->size(); ++idx) func(this->operator[](idx), idx);
        return *this;
    }

    template <class FuncType, _has_signature_enable_if<FuncType, void(reference, size_type, size_type)> = true,
              utl_mvl_require(dimension == Dimension::MATRIX)>
    self& for_each(FuncType func) {
        for (size_type idx = 0; idx < this->size(); ++idx) {
            const auto ij = this->get_ij_of_idx(idx);
            func(this->operator[](idx), ij.i, ij.j);
        }
        return *this;
    }

    template <class FuncType, _has_signature_enable_if<FuncType, value_type(const_reference)> = true,
              utl_mvl_require(ownership != Ownership::CONST_VIEW)>
    self& transform(FuncType func) {
        const auto func_wrapper = [&](reference elem) { elem = func(elem); };
        return this->for_each(func_wrapper);
    }

    template <class FuncType, _has_signature_enable_if<FuncType, value_type(const_reference, size_type)> = true,
              utl_mvl_require(dimension == Dimension::VECTOR && ownership != Ownership::CONST_VIEW)>
    self& transform(FuncType func) {
        const auto func_wrapper = [&](reference elem, size_type i) { elem = func(elem, i); };
        return this->for_each(func_wrapper);
    }

    template <class FuncType,
              _has_signature_enable_if<FuncType, value_type(const_reference, size_type, size_type)> = true,
              utl_mvl_require(dimension == Dimension::MATRIX && ownership != Ownership::CONST_VIEW)>
    self& transform(FuncType func) {
        const auto func_wrapper = [&](reference elem, size_type i, size_type j) { elem = func(elem, i, j); };
        return this->for_each(func_wrapper);
    }

    utl_mvl_reqs(ownership != Ownership::CONST_VIEW) self& fill(const_reference value) {
        for (size_type idx = 0; idx < this->size(); ++idx) this->operator[](idx) = value;
        return *this;
    }

    template <class FuncType, _has_signature_enable_if<FuncType, value_type()> = true,
              utl_mvl_require(ownership != Ownership::CONST_VIEW)>
    self& fill(FuncType func) {
        const auto func_wrapper = [&](reference elem) { elem = func(); };
        return this->for_each(func_wrapper);
    }

    template <class FuncType, _has_signature_enable_if<FuncType, value_type(size_type)> = true,
              utl_mvl_require(dimension == Dimension::VECTOR && ownership != Ownership::CONST_VIEW)>
    self& fill(FuncType func) {
        const auto func_wrapper = [&](reference elem, size_type i) { elem = func(i); };
        return this->for_each(func_wrapper);
    }

    template <class FuncType, _has_signature_enable_if<FuncType, value_type(size_type, size_type)> = true,
              utl_mvl_require(dimension == Dimension::MATRIX && ownership != Ownership::CONST_VIEW)>
    self& fill(FuncType func) {
        const auto func_wrapper = [&](reference elem, size_type i, size_type j) { elem = func(i, j); };
        return this->for_each(func_wrapper);
    }

    template <typename Compare, utl_mvl_require(ownership != Ownership::CONST_VIEW)>
    self& sort(Compare cmp) {
        std::sort(this->begin(), this->end(), cmp);
        return *this;
    }
    template <typename Compare, utl_mvl_require(ownership != Ownership::CONST_VIEW)>
    self& stable_sort(Compare cmp) {
        std::stable_sort(this->begin(), this->end(), cmp);
        return *this;
    }

    utl_mvl_reqs(ownership != Ownership::CONST_VIEW && _has_binary_op_less<value_type>::value) self& sort() {
        std::sort(this->begin(), this->end());
        return *this;
    }
    utl_mvl_reqs(ownership != Ownership::CONST_VIEW && _has_binary_op_less<value_type>::value) self& stable_sort() {
        std::stable_sort(this->begin(), this->end());
        return *this;
    }

    // --- Sparse Subviews ---
    // -----------------------

    // - Const views -
private:
    using _cref_triplet_array =
        std::vector<SparseEntry2D<std::reference_wrapper<const value_type>>>; // NOTE: Generalize for 1D

public:
    using sparse_const_view_type = GenericTensor<value_type, self::params::dimension, Type::SPARSE,
                                                 Ownership::CONST_VIEW, self::params::checking, Layout::SPARSE>;

    template <typename UnaryPredicate, _has_signature_enable_if<UnaryPredicate, bool(const_reference)> = true>
    [[nodiscard]] sparse_const_view_type filter(UnaryPredicate predicate) const {
        const auto forwarded_predicate = [&](const_reference elem, size_type, size_type) -> bool {
            return predicate(elem);
        };
        return this->filter(forwarded_predicate);
        // NOTE: This would need its own implementation for a proper 1D support
    }

    template <typename UnaryPredicate,
              _has_signature_enable_if<UnaryPredicate, bool(const_reference, size_type)> = true>
    [[nodiscard]] sparse_const_view_type filter(UnaryPredicate predicate) const {
        const auto forwarded_predicate = [&](const_reference elem, size_type i, size_type j) -> bool {
            const size_type idx = this->get_idx_of_ij(i, j);
            return predicate(elem, idx);
        };
        return this->filter(forwarded_predicate);
        // NOTE: This would need its own implementation for a proper 1D support
    }

    template <typename UnaryPredicate,
              _has_signature_enable_if<UnaryPredicate, bool(const_reference, size_type, size_type)> = true,
              utl_mvl_require(dimension == Dimension::MATRIX)>
    [[nodiscard]] sparse_const_view_type filter(UnaryPredicate predicate) const {
        // We can't preallocate triplets without scanning predicate through the whole matrix,
        // so we just push back entries into a vector and use to construct a sparse view
        _cref_triplet_array triplets;
        const auto          add_triplet_if_predicate = [&](const_reference elem, size_type i, size_type j) -> void {
            if (predicate(elem, i, j)) triplets.push_back({i, j, elem});
        };

        this->for_each(add_triplet_if_predicate);

        triplets.shrink_to_fit();
        return sparse_const_view_type(this->rows(), this->cols(), std::move(triplets));
    }

    utl_mvl_reqs(dimension == Dimension::MATRIX) [[nodiscard]] sparse_const_view_type diagonal() const {
        // Sparse matrices have no better way of getting a diagonal than filtering (i ==j)
        if constexpr (self::params::type == Type::SPARSE) {
            return this->filter([](const_reference, size_type i, size_type j) { return i == j; });
        }
        // Non-sparce matrices can just iterate over diagonal directly
        else {
            const size_type     min_size = std::min(this->rows(), this->cols());
            _cref_triplet_array triplets;
            triplets.reserve(min_size);
            for (size_type k = 0; k < min_size; ++k) triplets.push_back({k, k, this->operator()(k, k)});
            return sparse_const_view_type(this->rows(), this->cols(), std::move(triplets));
        }
    }

    // - Mutable views -
private:
    using _ref_triplet_array =
        std::vector<SparseEntry2D<std::reference_wrapper<value_type>>>; // NOTE: Generalize for 1D

public:
    using sparse_view_type = GenericTensor<value_type, self::params::dimension, Type::SPARSE, Ownership::VIEW,
                                           self::params::checking, Layout::SPARSE>;

    template <typename UnaryPredicate, utl_mvl_require(ownership != Ownership::CONST_VIEW),
              _has_signature_enable_if<UnaryPredicate, bool(const_reference)> = true>
    [[nodiscard]] sparse_view_type filter(UnaryPredicate predicate) {
        const auto forwarded_predicate = [&](const_reference elem, size_type, size_type) -> bool {
            return predicate(elem);
        };
        return this->filter(forwarded_predicate);
        // NOTE: This would need its own implementation for a proper 1D support
    }

    template <typename UnaryPredicate, utl_mvl_require(ownership != Ownership::CONST_VIEW),
              _has_signature_enable_if<UnaryPredicate, bool(const_reference, size_type)> = true>
    [[nodiscard]] sparse_view_type filter(UnaryPredicate predicate) {
        const auto forwarded_predicate = [&](const_reference elem, size_type i, size_type j) -> bool {
            const size_type idx = this->get_idx_of_ij(i, j);
            return predicate(elem, idx);
        };
        return this->filter(forwarded_predicate);
        // NOTE: This would need its own implementation for a proper 1D support
    }

    template <typename UnaryPredicate,
              _has_signature_enable_if<UnaryPredicate, bool(const_reference, size_type, size_type)> = true,
              utl_mvl_require(dimension == Dimension::MATRIX && ownership != Ownership::CONST_VIEW)>
    [[nodiscard]] sparse_view_type filter(UnaryPredicate predicate) {
        // This method implements actual filtering, others just forward predicates to it
        _ref_triplet_array triplets;
        // We can't preallocate triplets without scanning predicate through the whole matrix,
        // so we just push back entries into a vector and use to construct a sparse view
        const auto         add_triplet_if_predicate = [&](reference elem, size_type i, size_type j) -> void {
            if (predicate(elem, i, j)) triplets.push_back({i, j, elem});
        };

        this->for_each(add_triplet_if_predicate);

        triplets.shrink_to_fit();
        return sparse_view_type(this->rows(), this->cols(), std::move(triplets));
    }

    utl_mvl_reqs(dimension == Dimension::MATRIX && ownership != Ownership::CONST_VIEW) [[nodiscard]] sparse_view_type
        diagonal() {
        /* Sparse matrices have no better way of getting a diagonal than filtering (i == j) */
        if constexpr (self::params::type == Type::SPARSE) {
            return this->filter([](const_reference, size_type i, size_type j) { return i == j; });
        } /* Non-sparce matrices can just iterate over diagonal directly */
        else {
            const size_type    min_size = std::min(this->rows(), this->cols());
            _ref_triplet_array triplets;
            triplets.reserve(min_size);
            for (size_type k = 0; k < min_size; ++k) triplets.push_back({k, k, this->operator()(k, k)});
            return sparse_view_type(this->rows(), this->cols(), std::move(triplets));
        }
    }

    // --- Block Subviews ---
    // ----------------------
public:
    // - Const views -
    using block_const_view_type =
        std::conditional_t<self::params::type == Type::SPARSE, sparse_const_view_type,
                           GenericTensor<value_type, self::params::dimension, Type::STRIDED, Ownership::CONST_VIEW,
                                         self::params::checking, self::params::layout>>;

    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::SPARSE) [[nodiscard]] block_const_view_type
        block(size_type block_i, size_type block_j, size_type block_rows, size_type block_cols) const {
        // Sparse matrices have no better way of getting a block than filtering by { i, j }

        // Do the same thing as in .filter(), but shrink resulting view size to
        // { block_rows, block_cols } and shift indexation by { block_i, block_j }
        _cref_triplet_array triplets;

        const auto add_triplet_if_inside_block = [&](const_reference elem, size_type i, size_type j) -> void {
            if ((block_i <= i) && (i < block_i + block_rows) && (block_j <= j) && (j < block_j + block_cols))
                triplets.push_back({i - block_i, j - block_j, elem});
        };

        this->for_each(add_triplet_if_inside_block);

        triplets.shrink_to_fit();
        return block_const_view_type(block_rows, block_cols, std::move(triplets));
    }

    utl_mvl_reqs(dimension == Dimension::MATRIX && type != Type::SPARSE) [[nodiscard]] block_const_view_type
        block(size_type block_i, size_type block_j, size_type block_rows, size_type block_cols) const {
        if constexpr (self::params::layout == Layout::RC) {
            const size_type row_stride = this->row_stride() + this->col_stride() * (this->cols() - block_cols);
            const size_type col_stride = this->col_stride();
            return block_const_view_type(block_rows, block_cols, row_stride, col_stride,
                                         &this->operator()(block_i, block_j));
        }
        if constexpr (self::params::layout == Layout::CR) {
            const size_type row_stride = this->row_stride();
            const size_type col_stride = this->col_stride() + this->row_stride() * (this->rows() - block_rows);
            return block_const_view_type(block_rows, block_cols, row_stride, col_stride,
                                         &this->operator()(block_i, block_j));
        }
        _unreachable();
    }

    utl_mvl_reqs(dimension == Dimension::MATRIX) [[nodiscard]] block_const_view_type row(size_type i) const {
        return this->block(i, 0, 1, this->cols());
    }

    utl_mvl_reqs(dimension == Dimension::MATRIX) [[nodiscard]] block_const_view_type col(size_type j) const {
        return this->block(0, j, this->rows(), 1);
    }

    // - Mutable views -
    using block_view_type =
        std::conditional_t<self::params::type == Type::SPARSE, sparse_view_type,
                           GenericTensor<value_type, self::params::dimension, Type::STRIDED, Ownership::VIEW,
                                         self::params::checking, self::params::layout>>;

    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::SPARSE && ownership != Ownership::CONST_VIEW)
        [[nodiscard]] block_view_type
        block(size_type block_i, size_type block_j, size_type block_rows, size_type block_cols) {
        // Sparse matrices have no better way of getting a block than filtering by { i, j }

        // Do the same thing as in .filter(), but shrink resulting view size to
        // { block_rows, block_cols } and shift indexation by { block_i, block_j }
        _ref_triplet_array triplets;

        const auto add_triplet_if_inside_block = [&](reference elem, size_type i, size_type j) -> void {
            if ((block_i <= i) && (i < block_i + block_rows) && (block_j <= j) && (j < block_j + block_cols))
                triplets.push_back({i - block_i, j - block_j, elem});
        };

        this->for_each(add_triplet_if_inside_block);

        triplets.shrink_to_fit();
        return block_view_type(block_rows, block_cols, std::move(triplets));
    }

    utl_mvl_reqs(dimension == Dimension::MATRIX && type != Type::SPARSE && ownership != Ownership::CONST_VIEW)
        [[nodiscard]] block_view_type
        block(size_type block_i, size_type block_j, size_type block_rows, size_type block_cols) {
        if constexpr (self::params::layout == Layout::RC) {
            const size_type row_stride = this->row_stride() + this->col_stride() * (this->cols() - block_cols);
            const size_type col_stride = this->col_stride();
            return block_view_type(block_rows, block_cols, row_stride, col_stride, &this->operator()(block_i, block_j));
        }
        if constexpr (self::params::layout == Layout::CR) {
            const size_type row_stride = this->row_stride();
            const size_type col_stride = this->col_stride() + this->row_stride() * (this->rows() - block_rows);
            return block_view_type(block_rows, block_cols, row_stride, col_stride, &this->operator()(block_i, block_j));
        }
        _unreachable();
    }

    utl_mvl_reqs(dimension == Dimension::MATRIX && ownership != Ownership::CONST_VIEW) [[nodiscard]] block_view_type
        row(size_type i) {
        return this->block(i, 0, 1, this->cols());
    }

    utl_mvl_reqs(dimension == Dimension::MATRIX && ownership != Ownership::CONST_VIEW) [[nodiscard]] block_view_type
        col(size_type j) {
        return this->block(0, j, this->rows(), 1);
    }

    // --- Sparse operations ---
    // -------------------------

private:
    using _triplet_t = _choose_based_on_ownership<_ownership, SparseEntry2D<value_type>,
                                                  SparseEntry2D<std::reference_wrapper<value_type>>,
                                                  SparseEntry2D<std::reference_wrapper<const value_type>>>;

public:
    using sparse_entry_type = _triplet_t;

    utl_mvl_reqs(type == Type::SPARSE) [[nodiscard]] const std::vector<sparse_entry_type>& entries() const noexcept {
        return this->_data;
    }

    utl_mvl_reqs(type == Type::SPARSE && ownership != Ownership::CONST_VIEW)
        [[nodiscard]] std::vector<sparse_entry_type>& entries() noexcept {
        return this->_data;
    }

    utl_mvl_reqs(dimension == Dimension::MATRIX &&
                 type == Type::SPARSE) self& insert_triplets(const std::vector<sparse_entry_type>& triplets) {
        // Bulk-insert triplets and sort by index
        const auto ordering = [](const sparse_entry_type& l, const sparse_entry_type& r) -> bool {
            return (l.i < r.i) && (l.j < r.j);
        };

        this->_data.insert(this->_data.end(), triplets.begin(), triplets.end());
        std::sort(this->_data.begin(), this->_data.end(), ordering);

        return *this;
    }

    utl_mvl_reqs(dimension == Dimension::MATRIX &&
                 type == Type::SPARSE) self& rewrite_triplets(std::vector<sparse_entry_type>&& triplets) {
        // Move-construct all triplets at once and sort by index
        const auto ordering = [](const sparse_entry_type& l, const sparse_entry_type& r) -> bool {
            return (l.i < r.i) && (l.j < r.j);
        };

        this->_data = std::move(triplets);
        std::sort(this->_data.begin(), this->_data.end(), ordering);

        return *this;
    }

    utl_mvl_reqs(dimension == Dimension::MATRIX &&
                 type == Type::SPARSE) self& erase_triplets(std::vector<Index2D> indices) {
        // Erase triplets with {i, j} from 'indices' using the fact that both
        // 'indices' and triplets are sorted. We can scan through triplets once
        // while advancing 'cursor' when 'indices[cursor]' gets deleted, which
        // result in all necessary triplets being marked for erasure in order.
        std::sort(indices.begin(), indices.end());
        std::size_t cursor = 0;

        const auto erase_condition = [&](const sparse_entry_type& triplet) -> bool {
            /* Stop erasing once all target indices are handled */
            if (cursor == indices.size()) return false;
            if (indices[cursor].i == triplet.i && indices[cursor].j == triplet.j) {
                ++cursor;
                return true;
            }
            return false;
        };

        const auto iter = std::remove_if(this->_data.begin(), this->_data.end(), erase_condition);
        this->_data.erase(iter, this->_data.end());

        // Re-sort triplets just in case
        const auto ordering = [](const sparse_entry_type& l, const sparse_entry_type& r) -> bool {
            return (l.i < r.i) && (l.j < r.j);
        };
        std::sort(this->_data.begin(), this->_data.end(), ordering);

        return *this;
    }

    // --- Constructors ---
    // --------------------

    // - Matrix -
public:
    // Rule of five:
    // copy-ctor       - [+] (deduced from copy-assignment)
    // move-ctor       - [+] (= default)
    // copy-assignment - [+] (custom for dense/strided, same as default for sparse)
    // move-assignment - [+] (= default)
    // destructor      - [+] (= default)

    // Move-ctor is default for all types
    GenericTensor(self&& other) noexcept = default;

    // Move-assignment is default for all types
    self& operator=(self&& other) noexcept = default;

    // Copy-ctor is deduced from assignment operator
    GenericTensor(const self& other) { *this = other; }

    // Copy-assignment
    self& operator=(const self& other) {
        // Note: copy-assignment operator CANNOT be templated, it has to be implemented with 'if constexpr'
        this->_rows = other.rows();
        this->_cols = other.cols();
        if constexpr (self::params::type == Type::DENSE) {
            this->_data = std::move(_make_unique_ptr_array<value_type>(this->size()));
            std::copy(other.begin(), other.end(), this->begin());
        }
        if constexpr (self::params::type == Type::STRIDED) {
            this->_row_stride = other.row_stride();
            this->_col_stride = other.col_stride();
            this->_data       = std::move(_make_unique_ptr_array<value_type>(this->size()));
            std::copy(other.begin(), other.end(), this->begin());
        }
        if constexpr (self::params::type == Type::SPARSE) { this->_data = other._data; }
        return *this;
    }

    // Default-ctor (containers)
    utl_mvl_reqs(ownership == Ownership::CONTAINER) GenericTensor() noexcept {}

    // Default-ctor (views)
    utl_mvl_reqs(ownership != Ownership::CONTAINER) GenericTensor() noexcept = delete;

    // Copy-assignment over the config boundaries
    // We can change checking config, copy from matrices with different layouts,
    // copy from views and even matrices of other types
    template <Type other_type, Ownership other_ownership, Checking other_checking, Layout other_layout,
              utl_mvl_require(dimension == Dimension::MATRIX && type == Type::DENSE &&
                              ownership == Ownership::CONTAINER)>
    self& operator=(const GenericTensor<value_type, self::params::dimension, other_type, other_ownership,
                                        other_checking, other_layout>& other) {
        this->_rows = other.rows();
        this->_cols = other.cols();
        this->_data = std::move(_make_unique_ptr_array<value_type>(this->size()));
        this->fill(value_type());
        other.for_each([&](const value_type& element, size_type i, size_type j) { this->operator()(i, j) = element; });
        return *this;
        // copying from sparse to dense works, all elements that weren't in the sparse matrix remain default-initialized
    }

    template <Type other_type, Ownership other_ownership, Checking other_checking, Layout other_layout,
              utl_mvl_require(dimension == Dimension::MATRIX && type == Type::STRIDED &&
                              ownership == Ownership::CONTAINER)>
    self& operator=(const GenericTensor<value_type, self::params::dimension, other_type, other_ownership,
                                        other_checking, other_layout>& other) {
        this->_rows       = other.rows();
        this->_cols       = other.cols();
        this->_row_stride = other.row_stride();
        this->_col_strude = other.col_stride();
        this->_data       = std::move(_make_unique_ptr_array<value_type>(this->size()));
        this->fill(value_type());
        // Not quite sure whether swapping strides when changing layouts like this is okay,
        // but it seems to be correct
        if constexpr (self::params::layout != other_layout) std::swap(this->_row_stride, this->_cols_stride);
        other.for_each([&](const value_type& element, size_type i, size_type j) { this->operator()(i, j) = element; });
        return *this;
        // copying from sparse to strided works, all elements that weren't in the sparse matrix remain
        // default-initialized
    }

    template <Type other_type, Ownership other_ownership, Checking other_checking, Layout other_layout,
              utl_mvl_require(dimension == Dimension::MATRIX && type == Type::SPARSE &&
                              ownership == Ownership::CONTAINER)>
    self& operator=(const GenericTensor<value_type, self::params::dimension, other_type, other_ownership,
                                        other_checking, other_layout>& other) {
        this->_rows = other.rows();
        this->_cols = other.cols();
        std::vector<sparse_entry_type> triplets;

        // Other sparse matrices can be trivially copied
        if constexpr (other_type == Type::SPARSE) {
            triplets.reserve(other.size());
            other.for_each([&](const value_type& elem, size_type i, size_type j) { triplets.push_back({i, j, elem}); });
        }
        // Non-sparse matrices are filtered by non-default-initialized-elements to construct a sparse subset
        else {
            other.for_each([&](const_reference elem, size_type i, size_type j) {
                if (elem != value_type()) triplets.push_back({i, j, elem});
            });
        }

        this->rewrite_triplets(std::move(triplets));
        return *this;
    }

    // Copy-ctor over the config boundaries (deduced from assignment over config boundaries)
    template <Type other_type, Ownership other_ownership, Checking other_checking, Layout other_layout,
              utl_mvl_require(dimension == Dimension::MATRIX && ownership == Ownership::CONTAINER)>
    GenericTensor(const GenericTensor<value_type, self::params::dimension, other_type, other_ownership, other_checking,
                                      other_layout>& other) {
        *this = other;
    }

    // Move-assignment over config boundaries
    // Note: Unlike copying, we can't change layout, only checking config
    // Also 'other' can no longer be a view or have a different type
    template <Checking other_checking, utl_mvl_require(dimension == Dimension::MATRIX && type == Type::DENSE &&
                                                       ownership == Ownership::CONTAINER)>
    self& operator=(GenericTensor<value_type, self::params::dimension, self::params::type, self::params::ownership,
                                  other_checking, self::params::layout>&& other) {
        this->_rows = other.rows();
        this->_cols = other.cols();
        this->_data = std::move(other._data);
        return *this;
    }

    template <Checking other_checking, utl_mvl_require(dimension == Dimension::MATRIX && type == Type::STRIDED &&
                                                       ownership == Ownership::CONTAINER)>
    self& operator=(GenericTensor<value_type, self::params::dimension, self::params::type, self::params::ownership,
                                  other_checking, self::params::layout>&& other) {
        this->_rows       = other.rows();
        this->_cols       = other.cols();
        this->_row_stride = other.row_stride();
        this->_col_stride = other.col_stride();
        this->_data       = std::move(other._data);
        return *this;
    }

    template <Checking other_checking, utl_mvl_require(dimension == Dimension::MATRIX && type == Type::SPARSE &&
                                                       ownership == Ownership::CONTAINER)>
    self& operator=(GenericTensor<value_type, self::params::dimension, self::params::type, self::params::ownership,
                                  other_checking, self::params::layout>&& other) {
        this->_rows = other.rows();
        this->_cols = other.cols();
        this->_data = std::move(other._data);
        return *this;
    }

    // Move-ctor over the config boundaries (deduced from move-assignment over config boundaries)
    template <Type other_type, Ownership other_ownership, Checking other_checking, Layout other_layout,
              utl_mvl_require(dimension == Dimension::MATRIX && ownership == Ownership::CONTAINER)>
    GenericTensor(GenericTensor<value_type, self::params::dimension, other_type, other_ownership, other_checking,
                                other_layout>&& other) {
        *this = std::move(other);
    }

    // Init-with-value
    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::DENSE &&
                 ownership ==
                     Ownership::CONTAINER) explicit GenericTensor(size_type rows, size_type cols,
                                                                  const_reference value = value_type()) noexcept {
        this->_rows = rows;
        this->_cols = cols;
        this->_data = std::move(_make_unique_ptr_array<value_type>(this->size()));
        this->fill(value);
    }

    // Init-with-lambda
    template <typename FuncType, utl_mvl_require(dimension == Dimension::MATRIX && type == Type::DENSE &&
                                                 ownership == Ownership::CONTAINER)>
    explicit GenericTensor(size_type rows, size_type cols, FuncType init_func) {
        // .fill() already takes care of preventing improper values of 'FuncType', no need to do the check here
        this->_rows = rows;
        this->_cols = cols;
        this->_data = std::move(_make_unique_ptr_array<value_type>(this->size()));
        this->fill(init_func);
    }

    // Init-with-ilist
    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::DENSE && ownership == Ownership::CONTAINER)
        GenericTensor(std::initializer_list<std::initializer_list<value_type>> init) {
        this->_rows = init.size();
        this->_cols = (*init.begin()).size();
        this->_data = std::move(_make_unique_ptr_array<value_type>(this->size()));

        // Check dimensions (throw if cols have different dimensions)
        for (auto row_it = init.begin(); row_it < init.end(); ++row_it)
            if (static_cast<size_type>((*row_it).end() - (*row_it).begin()) != this->cols())
                throw std::invalid_argument("Initializer list dimensions don't match.");

        // Copy elements
        for (size_type i = 0; i < this->rows(); ++i)
            for (size_type j = 0; j < this->cols(); ++j) this->operator()(i, j) = (init.begin()[i]).begin()[j];
    }

    // Init-with-data
    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::DENSE &&
                 ownership == Ownership::CONTAINER) explicit GenericTensor(size_type rows, size_type cols,
                                                                           pointer data_ptr) noexcept {
        this->_rows = rows;
        this->_cols = cols;
        this->_data = std::move(decltype(this->_data)(data_ptr));
    }

    // - Matrix View -

    // Init-from-data
    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::DENSE &&
                 ownership == Ownership::VIEW) explicit GenericTensor(size_type rows, size_type cols,
                                                                      pointer data_ptr) {
        this->_rows = rows;
        this->_cols = cols;
        this->_data = data_ptr;
    }

    // Init-from-tensor (any tensor of the same API type)
    template <Ownership other_ownership, Checking other_checking, Layout other_layout,
              utl_mvl_require(dimension == Dimension::MATRIX && type == Type::DENSE && ownership == Ownership::VIEW)>
    GenericTensor(GenericTensor<value_type, self::params::dimension, self::params::type, other_ownership,
                                other_checking, other_layout>& other) {
        this->_rows = other.rows();
        this->_cols = other.cols();
        this->_data = other.data();
    }

    // - Const Matrix View -

    // Init-from-data
    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::DENSE &&
                 ownership == Ownership::CONST_VIEW) explicit GenericTensor(size_type rows, size_type cols,
                                                                            const_pointer data_ptr) {
        this->_rows = rows;
        this->_cols = cols;
        this->_data = data_ptr;
    }

    // Init-from-tensor (any tensor of the same API type)
    template <Ownership other_ownership, Checking other_checking, Layout other_layout,
              utl_mvl_require(dimension == Dimension::MATRIX && type == Type::DENSE &&
                              ownership == Ownership::CONST_VIEW)>
    GenericTensor(const GenericTensor<value_type, self::params::dimension, self::params::type, other_ownership,
                                      other_checking, other_layout>& other) {
        this->_rows = other.rows();
        this->_cols = other.cols();
        this->_data = other.data();
    }

    // - Strided Matrix -

    // Init-with-value
    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::STRIDED &&
                 ownership ==
                     Ownership::CONTAINER) explicit GenericTensor(size_type rows, size_type cols, size_type row_stride,
                                                                  size_type       col_stride,
                                                                  const_reference value = value_type()) noexcept {
        this->_rows       = rows;
        this->_cols       = cols;
        this->_row_stride = row_stride;
        this->_col_stride = col_stride;
        // Allocates size is NOT the same as .size() due to padding, see notes on '_total_allocated_size()'
        this->_data       = std::move(_make_unique_ptr_array<value_type>(this->_total_allocated_size()));
        this->fill(value);
    }

    // Init-with-lambda
    template <typename FuncType, utl_mvl_require(dimension == Dimension::MATRIX && type == Type::STRIDED &&
                                                 ownership == Ownership::CONTAINER)>
    explicit GenericTensor(size_type rows, size_type cols, size_type row_stride, size_type col_stride,
                           FuncType init_func) {
        // .fill() already takes care of preventing improper values of 'FuncType', no need to do the check here
        this->_rows       = rows;
        this->_cols       = cols;
        this->_row_stride = row_stride;
        this->_col_stride = col_stride;
        // Allocates size is NOT the same as .size() due to padding, see notes on '_total_allocated_size()'
        this->_data       = std::move(_make_unique_ptr_array<value_type>(this->_total_allocated_size()));
        this->fill(init_func);
    }

    // Init-with-ilist
    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::STRIDED && ownership == Ownership::CONTAINER)
        GenericTensor(std::initializer_list<std::initializer_list<value_type>> init, size_type row_stride,
                      size_type col_stride) {
        this->_rows       = init.size();
        this->_cols       = (*init.begin()).size();
        this->_row_stride = row_stride;
        this->_col_stride = col_stride;
        // Allocates size is NOT the same as .size() due to padding, see notes on '_total_allocated_size()'
        this->_data       = std::move(_make_unique_ptr_array<value_type>(this->_total_allocated_size()));

        // Check dimensions (throw if cols have different dimensions)
        for (auto row_it = init.begin(); row_it < init.end(); ++row_it)
            if ((*row_it).end() - (*row_it).begin() != this->_cols)
                throw std::invalid_argument("Initializer list dimensions don't match.");

        // Copy elements
        for (size_type i = 0; i < this->rows(); ++i)
            for (size_type j = 0; j < this->cols(); ++j) this->operator()(i, j) = (init.begin()[i]).begin()[j];
    }

    // Init-with-data
    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::STRIDED &&
                 ownership == Ownership::CONTAINER) explicit GenericTensor(size_type rows, size_type cols,
                                                                           size_type row_stride, size_type col_stride,
                                                                           pointer data_ptr) noexcept {
        this->_rows       = rows;
        this->_cols       = cols;
        this->_row_stride = row_stride;
        this->_col_stride = col_stride;
        this->_data       = std::move(decltype(this->_data)(data_ptr));
    }

    // - Strided Matrix View -

    // Init-from-data
    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::STRIDED &&
                 ownership == Ownership::VIEW) explicit GenericTensor(size_type rows, size_type cols,
                                                                      size_type row_stride, size_type col_stride,
                                                                      pointer data_ptr) {
        this->_rows       = rows;
        this->_cols       = cols;
        this->_row_stride = row_stride;
        this->_col_stride = col_stride;
        this->_data       = data_ptr;
    }

    // Init-from-tensor (any tensor of the same API type)
    template <Ownership other_ownership, Checking other_checking, Layout other_layout,
              utl_mvl_require(dimension == Dimension::MATRIX && type == Type::STRIDED && ownership == Ownership::VIEW)>
    GenericTensor(GenericTensor<value_type, self::params::dimension, self::params::type, other_ownership,
                                other_checking, other_layout>& other) {
        this->_rows       = other.rows();
        this->_cols       = other.cols();
        this->_row_stride = other.row_stride();
        this->_col_stride = other.col_stride();
        this->_data       = other.data();
    }

    // - Const Strided Matrix View -

    // Init-from-data
    utl_mvl_reqs(dimension == Dimension::MATRIX && type == Type::STRIDED &&
                 ownership == Ownership::CONST_VIEW) explicit GenericTensor(size_type rows, size_type cols,
                                                                            size_type row_stride, size_type col_stride,
                                                                            const_pointer data_ptr) {
        this->_rows       = rows;
        this->_cols       = cols;
        this->_row_stride = row_stride;
        this->_col_stride = col_stride;
        this->_data       = data_ptr;
    }

    // Init-from-tensor (any tensor of the same API type)
    template <Ownership other_ownership, Checking other_checking, Layout other_layout,
              utl_mvl_require(dimension == Dimension::MATRIX && type == Type::STRIDED &&
                              ownership == Ownership::CONST_VIEW)>
    GenericTensor(const GenericTensor<value_type, self::params::dimension, self::params::type, other_ownership,
                                      other_checking, other_layout>& other) {
        this->_rows       = other.rows();
        this->_cols       = other.cols();
        this->_row_stride = other.row_stride();
        this->_col_stride = other.col_stride();
        this->_data       = other.data();
    }

    // - Sparse Matrix / Sparse Matrix View / Sparse Matrix Const View -

    // Init-from-data (copy)
    utl_mvl_reqs(dimension == Dimension::MATRIX &&
                 type == Type::SPARSE) explicit GenericTensor(size_type rows, size_type cols,
                                                              const std::vector<sparse_entry_type>& data) {
        this->_rows = rows;
        this->_cols = cols;
        this->insert_triplets(std::move(data));
    }

    // Init-from-data (move)
    utl_mvl_reqs(dimension == Dimension::MATRIX &&
                 type == Type::SPARSE) explicit GenericTensor(size_type rows, size_type cols,
                                                              std::vector<sparse_entry_type>&& data) {
        this->_rows = rows;
        this->_cols = cols;
        this->rewrite_triplets(std::move(data));
    }
};

// ===========================
// --- Predefined Typedefs ---
// ===========================

constexpr auto _default_checking        = Checking::NONE;
constexpr auto _default_layout_dense_2d = Layout::RC;

// - Dense 2D -
template <typename T, Checking checking = _default_checking, Layout layout = _default_layout_dense_2d>
using Matrix = GenericTensor<T, Dimension::MATRIX, Type::DENSE, Ownership::CONTAINER, checking, layout>;

template <typename T, Checking checking = _default_checking, Layout layout = _default_layout_dense_2d>
using MatrixView = GenericTensor<T, Dimension::MATRIX, Type::DENSE, Ownership::VIEW, checking, layout>;

template <typename T, Checking checking = _default_checking, Layout layout = _default_layout_dense_2d>
using ConstMatrixView = GenericTensor<T, Dimension::MATRIX, Type::DENSE, Ownership::CONST_VIEW, checking, layout>;

// - Strided 2D -
template <typename T, Checking checking = _default_checking, Layout layout = _default_layout_dense_2d>
using StridedMatrix = GenericTensor<T, Dimension::MATRIX, Type::STRIDED, Ownership::CONTAINER, checking, layout>;

template <typename T, Checking checking = _default_checking, Layout layout = _default_layout_dense_2d>
using StridedMatrixView = GenericTensor<T, Dimension::MATRIX, Type::STRIDED, Ownership::VIEW, checking, layout>;

template <typename T, Checking checking = _default_checking, Layout layout = _default_layout_dense_2d>
using ConstStridedMatrixView =
    GenericTensor<T, Dimension::MATRIX, Type::STRIDED, Ownership::CONST_VIEW, checking, layout>;

// - Sparse 2D -
template <typename T, Checking checking = _default_checking>
using SparseMatrix = GenericTensor<T, Dimension::MATRIX, Type::SPARSE, Ownership::CONTAINER, checking, Layout::SPARSE>;

template <typename T, Checking checking = _default_checking>
using SparseMatrixView = GenericTensor<T, Dimension::MATRIX, Type::SPARSE, Ownership::VIEW, checking, Layout::SPARSE>;

template <typename T, Checking checking = _default_checking>
using ConstSparseMatrixView =
    GenericTensor<T, Dimension::MATRIX, Type::SPARSE, Ownership::CONST_VIEW, checking, Layout::SPARSE>;

// ==================
// --- Formatters ---
// ==================

namespace format {

// --- Implementation ---
// ----------------------

// The "header" of all human-readable formats that displays
// some meta info with tensor type and dimensions.
template <utl_mvl_tensor_arg_defs>
[[nodiscard]] std::string _tensor_meta_string(const GenericTensor<utl_mvl_tensor_arg_vals>& tensor) {
    std::string buffer;

    if constexpr (_type == Type::DENSE) buffer += "Dense";
    if constexpr (_type == Type::STRIDED) buffer += "Strided";
    if constexpr (_type == Type::SPARSE) buffer += "Sparse";
    if constexpr (_dimension == Dimension::VECTOR) buffer += stringify(" vector [size = ", tensor.size(), "]:\n");
    if constexpr (_dimension == Dimension::MATRIX)
        buffer += stringify(" matrix [size = ", tensor.size(), "] (", tensor.rows(), " x ", tensor.cols(), "):\n");

    return buffer;
}

// Human-readable formats automatically collapse matrices
// that are too  large to be reasonably parsed by a human
constexpr std::size_t _max_displayed_flat_size = 30 * 30;

template <utl_mvl_tensor_arg_defs>
[[nodiscard]] std::string _as_too_large(const GenericTensor<utl_mvl_tensor_arg_vals>& tensor) {
    return stringify(_tensor_meta_string(tensor), "  <hidden due to large size>\n");
}

// Generic method to do "dense matrix print" with given delimers.
// Cuts down on repitition since a lot of formats only differ in the delimers used.
template <class T, Type type, Ownership ownership, Checking checking, Layout layout, class Func>
[[nodiscard]] std::string
_generic_dense_format(const GenericTensor<T, Dimension::MATRIX, type, ownership, checking, layout>& tensor,      //
                      std::string_view                                                              begin,       //
                      std::string_view                                                              row_begin,   //
                      std::string_view                                                              row_delimer, //
                      std::string_view                                                              row_end,     //
                      std::string_view                                                              end,         //
                      Func                                                                          stringifier  //
) {
    if (tensor.empty()) return (std::string() += begin) += end;

    Matrix<std::string> strings(tensor.rows(), tensor.cols());

    // Stringify
    if constexpr (type == Type::SPARSE) strings.fill("-");
    tensor.for_each([&](const T& elem, std::size_t i, std::size_t j) { strings(i, j) = stringifier(elem); });
    // this takes care of sparsity, if the matrix is sparse we prefill 'strings' with "-" and then fill appropriate
    // {i, j} with actual stringified values from the tensor. For dense matrices no unnecessary work is done.

    // Get column widths - we want matrix to format nice and aligned
    std::vector<std::size_t> column_widths(strings.cols(), 0);
    for (std::size_t i = 0; i < strings.rows(); ++i)
        for (std::size_t j = 0; j < strings.cols(); ++j)
            column_widths[j] = std::max(column_widths[j], strings(i, j).size());

    // Format with proper alignment
    std::string buffer(begin);
    for (std::size_t i = 0; i < strings.rows(); ++i) {
        buffer += row_begin;
        for (std::size_t j = 0; j < strings.cols(); ++j) {
            if (strings(i, j).size() < column_widths[j]) buffer.append(column_widths[j] - strings(i, j).size(), ' ');
            buffer += strings(i, j);
            if (j + 1 < strings.cols()) buffer += row_delimer;
        }
        buffer += row_end;
    }
    buffer += end;

    return buffer;
}

// --- Human-readable formats ---
// ------------------------------

template <utl_mvl_tensor_arg_defs, class Func = default_stringifier<T>>
[[nodiscard]] std::string as_vector(const GenericTensor<utl_mvl_tensor_arg_vals>& tensor, Func stringifier = Func()) {
    if (tensor.size() > _max_displayed_flat_size) return _as_too_large(tensor);

    std::string buffer = _tensor_meta_string(tensor);

    buffer += "  { ";
    for (std::size_t idx = 0; idx < tensor.size(); ++idx) {
        buffer += stringifier(tensor[idx]);
        if (idx + 1 < tensor.size()) buffer += ", ";
    }
    buffer += " }\n";

    return buffer;
}

template <utl_mvl_tensor_arg_defs, class Func = default_stringifier<T>>
[[nodiscard]] std::string as_dictionary(const GenericTensor<utl_mvl_tensor_arg_vals>& tensor,
                                        Func                                          stringifier = Func()) {
    if (tensor.size() > _max_displayed_flat_size) return _as_too_large(tensor);

    std::string buffer = _tensor_meta_string(tensor);

    if constexpr (_dimension == Dimension::MATRIX) {
        tensor.for_each([&](const T& elem, std::size_t i, std::size_t j) {
            buffer += stringify("(", i, ", ", j, ") = ");
            buffer += stringifier(elem);
            buffer += '\n';
        });
    } else {
        tensor.for_each([&](const T& elem, std::size_t i) {
            buffer += stringify("(", i, ") = ");
            buffer += stringifier(elem);
            buffer += '\n';
        });
    }

    return buffer;
}

template <utl_mvl_tensor_arg_defs, class Func = default_stringifier<T>>
[[nodiscard]] std::string as_matrix(const GenericTensor<utl_mvl_tensor_arg_vals>& tensor, Func stringifier = Func()) {
    if (tensor.size() > _max_displayed_flat_size) return _as_too_large(tensor);

    return _generic_dense_format(tensor, _tensor_meta_string(tensor), "  [ ", " ", " ]\n", "", stringifier);
}

// --- Export formats ---
// ----------------------

template <utl_mvl_tensor_arg_defs, class Func = default_stringifier<T>>
[[nodiscard]] std::string as_raw(const GenericTensor<utl_mvl_tensor_arg_vals>& tensor, Func stringifier = Func()) {
    return _generic_dense_format(tensor, "", "", " ", "\n", "", stringifier);
}

template <utl_mvl_tensor_arg_defs, class Func = default_stringifier<T>>
[[nodiscard]] std::string as_csv(const GenericTensor<utl_mvl_tensor_arg_vals>& tensor, Func stringifier = Func()) {
    return _generic_dense_format(tensor, "", "", ", ", "\n", "", stringifier);
}

template <utl_mvl_tensor_arg_defs, class Func = default_stringifier<T>>
[[nodiscard]] std::string as_json(const GenericTensor<utl_mvl_tensor_arg_vals>& tensor, Func stringifier = Func()) {
    return _generic_dense_format(tensor, "[\n", "    [ ", ", ", " ]\n", "]\n", stringifier);
}

template <utl_mvl_tensor_arg_defs, class Func = default_stringifier<T>>
[[nodiscard]] std::string as_latex(const GenericTensor<utl_mvl_tensor_arg_vals>& tensor, Func stringifier = Func()) {
    return _generic_dense_format(tensor, "\\begin{pmatrix}\n", "  ", " & ", " \\\\\n", "\\end{pmatrix}\n", stringifier);
}


} // namespace format

// ================================
// --- Linear algebra operators ---
// ================================

// --- Unary operator implementation ----
// --------------------------------------

template <class L, class Op,                                                    //
          _is_tensor_enable_if<L> = true,                                       //
          class return_type       = typename std::decay_t<L>::owning_reflection //
          >
return_type apply_unary_op(L&& left, Op&& op) {
    using reference = typename std::decay_t<L>::reference;

    // Reuse r-value if possible
    return_type res = std::forward<L>(left);

    // '.for_each()' takes care of possible sparsity
    res.for_each([&](reference elem) { elem = op(std::move(elem)); });

    return res;
}

// --- Unary operator API ----
// ---------------------------

// std <functional> has function objects for every operator possible,
// except unary '+' for some reason, which is why we have to implement it ourselves.
// This is the exact same thing as 'std::negate<>' but for operator '+'.
template <class T>
struct _unary_plus_functor {
    constexpr T operator()(const T& lhs) const { return +lhs; }
};

template <class L, _is_tensor_enable_if<L> = true, class value_type = typename std::decay_t<L>::value_type,
          _has_unary_op_plus_enable_if<value_type> = true>
auto operator+(L&& left) {
    return apply_unary_op(std::forward<L>(left), _unary_plus_functor<value_type>());
}

template <class L, _is_tensor_enable_if<L> = true, class value_type = typename std::decay_t<L>::value_type,
          _has_unary_op_minus_enable_if<value_type> = true>
auto operator-(L&& left) {
    return apply_unary_op(std::forward<L>(left), std::negate<value_type>());
}

// --- Binary operator implementation ---
// --------------------------------------

// Doing things "in a dumb but simple way" would be to just have all operators take arguments as const-refs and
// return a copy, however we can speed things up a lot by properly using perfect forwarding, which would reuse
// r-lvalues if possible to avoid allocation. Doing so would effectively change something like this:
//    res = A + B - C - D + E
// from 5 (!) copies to only 1, since first operator will create an r-value that gets propagated and reused by
// all others. This however introduces it's own set of challenges since instead of traditional overloading we
// now have to resort to SFINAE-driven imitation of it and carefully watch what we restrict.
//
// So here are the key points of implementing all of this:
//
//    1. All unary and binary operators ('-', '+', '*' and etc.) can benefit from reusing r-values and moving
//       one of the arguments into the result rather than copy. This can only happen to CONTAINER args that
//       correspond to the return type.
//
//    2. Binary operators have following return types for different tensors:
//       - (1)  dense +  dense =>  dense      (complexity O(N^2))
//       - (2)  dense + sparse =>  dense      (complexity O(N)  )
//       - (3) sparse +  dense =>  dense      (complexity O(N)  )
//       - (4) sparse + sparse => sparse      (complexity O(N)  )
//       return types inherit option arguments from the lhs and always have 'ownership == CONTAINER',
//       while argument can be anything, including views. To implement the list above efficiently there
//       is no other way but to make 4 separate implementation, tailored for each level of sparsity.
//
//    3. All binary operators can be written in a generic form like 'apply_binary_operator(A, B, op)'
//       and specialized with operator functors froms std <functional>. All we need for individual
//       operators is to figure out correct boilerplate with perfect forwarding. "Overload resolution"
//       will be performed by the 'apply_operator' method.
//
// It's a good question whether to threat binary '*' as element-wise product (which would be in line with other
// operators) or a matrix product, but in the end it seems doing it element-wise would be too confusing for the
// user, so we leave '*' as a matrix product and declate element-wise product as a function 'elementwise_product()'
// since no existing operators seem suitable for such overload.

// (1)  dense +  dense =>  dense
template <class L, class R, class Op,                                                                     //
          _are_tensors_with_same_value_type_enable_if<L, R> = true,                                       //
          _is_nonsparse_tensor_enable_if<L>                 = true,                                       //
          _is_nonsparse_tensor_enable_if<R>                 = true,                                       //
          class return_type                                 = typename std::decay_t<L>::owning_reflection //
          >
return_type apply_binary_op(L&& left, R&& right, Op&& op) {
    utl_mvl_assert(left.rows() == right.rows());
    utl_mvl_assert(left.cols() == right.cols());

    using reference = typename std::decay_t<L>::reference;
    using size_type = typename std::decay_t<L>::size_type;

    return_type res;

    // Reuse r-value arguments if possible while preserving the order of operations
    if constexpr (std::is_rvalue_reference_v<L> && std::decay_t<L>::params::ownership == Ownership::CONTAINER) {
        res = std::forward<L>(left);
        res.for_each([&](reference elem, size_type i, size_type j) { elem = op(std::move(elem), right(i, j)); });
    } else if constexpr (std::is_rvalue_reference_v<R> && std::decay_t<R>::params::ownership == Ownership::CONTAINER) {
        res = std::forward<R>(right);
        res.for_each([&](reference elem, size_type i, size_type j) { elem = op(left(i, j), std::move(elem)); });
    } else {
        res = return_type(left.rows(), left.cols());
        res.for_each([&](reference elem, size_type i, size_type j) { elem = op(left(i, j), right(i, j)); });
    }

    return res;
}

// (2)  dense + sparse =>  dense
template <class L, class R, class Op,                                                                     //
          _are_tensors_with_same_value_type_enable_if<L, R> = true,                                       //
          _is_nonsparse_tensor_enable_if<L>                 = true,                                       //
          _is_sparse_tensor_enable_if<R>                    = true,                                       //
          class return_type                                 = typename std::decay_t<L>::owning_reflection //
          >
return_type apply_binary_op(L&& left, R&& right, Op&& op) {
    utl_mvl_assert(left.rows() == right.rows());
    utl_mvl_assert(left.cols() == right.cols());

    using reference = typename std::decay_t<L>::reference;
    using size_type = typename std::decay_t<L>::size_type;

    // Reuse r-value if possible
    return_type res = std::forward<L>(left);

    // '.for_each()' to only iterate sparse elements
    right.for_each([&](reference elem, size_type i, size_type j) { res(i, j) = op(std::move(res(i, j)), elem); });

    return res;
}

// (3) sparse +  dense =>  dense
template <class L, class R, class Op,                                                                     //
          _are_tensors_with_same_value_type_enable_if<L, R> = true,                                       //
          _is_sparse_tensor_enable_if<L>                    = true,                                       //
          _is_nonsparse_tensor_enable_if<R>                 = true,                                       //
          class return_type                                 = typename std::decay_t<R>::owning_reflection //
          >
return_type apply_binary_op(L&& left, R&& right, Op&& op) {
    utl_mvl_assert(left.rows() == right.rows());
    utl_mvl_assert(left.cols() == right.cols());

    using reference = typename std::decay_t<R>::reference;
    using size_type = typename std::decay_t<R>::size_type;

    // Reuse r-value if possible
    return_type res = std::forward<R>(right);

    // '.for_each()' to only iterate sparse elements
    left.for_each([&](reference elem, size_type i, size_type j) { res(i, j) = op(elem, std::move(res(i, j))); });

    return res;
}

// (4) sparse + sparse => sparse
template <class L, class R, class Op,                                                                     //
          _are_tensors_with_same_value_type_enable_if<L, R> = true,                                       //
          _is_sparse_tensor_enable_if<L>                    = true,                                       //
          _is_sparse_tensor_enable_if<R>                    = true,                                       //
          class return_type                                 = typename std::decay_t<L>::owning_reflection //
          >
return_type apply_binary_op(L&& left, R&& right, Op&& op) {
    utl_mvl_assert(left.rows() == right.rows());
    utl_mvl_assert(left.cols() == right.cols());

    using sparse_entry = typename std::decay_t<L>::sparse_entry_type;
    using value_type   = typename std::decay_t<L>::value_type;

    std::vector<sparse_entry> res_triplets;
    res_triplets.reserve(std::max(left.size(), right.size()));
    // not enough when matrices have different sparsity patterns, but good enough for initial guess

    std::size_t i = 0, j = 0;

    // Merge sparsity patterns
    while (i < left.size() && j < right.size()) {
        // Entry present only in lhs sparsity pattern
        if (left._data[i] < right._data[j]) {
            res_triplets.emplace_back(
                _apply_binary_op_to_sparse_entry_and_value(std::forward<L>(left)._data[i++], value_type{}, op));
        }
        // Entry present only in rhs sparsity pattern
        else if (left._data[i] > right._data[j]) {
            res_triplets.emplace_back(
                _apply_binary_op_to_value_and_sparse_entry(value_type{}, std::forward<R>(right)._data[j++], op));
        }
        // Entry present in both sparsity patterns
        else {
            res_triplets.emplace_back(_apply_binary_op_to_sparse_entries(std::forward<L>(left)._data[i++],
                                                                         std::forward<R>(right)._data[j++], op));
        }
    }
    // Copy the rest of lhs (if needed)
    while (i < left.size()) res_triplets.emplace_back(std::forward<L>(left)._data[i++]);
    // Copy the rest of rhs (if needed)
    while (j < right.size()) res_triplets.emplace_back(std::forward<R>(right)._data[j++]);

    return return_type(left.rows(), left.cols(), std::move(res_triplets));
}

// --- Binary operator API ---
// ---------------------------

template <class L, class R, _are_tensors_with_same_value_type_enable_if<L, R> = true,
          class value_type = typename std::decay_t<L>::value_type, _has_binary_op_plus_enable_if<value_type> = true>
auto operator+(L&& left, R&& right) {
    return apply_binary_op(std::forward<L>(left), std::forward<R>(right), std::plus<value_type>());
}

template <class L, class R, _are_tensors_with_same_value_type_enable_if<L, R> = true,
          class value_type = typename std::decay_t<L>::value_type, _has_binary_op_minus_enable_if<value_type> = true>
auto operator-(L&& left, R&& right) {
    return apply_binary_op(std::forward<L>(left), std::forward<R>(right), std::minus<value_type>());
}

template <class L, class R, _are_tensors_with_same_value_type_enable_if<L, R> = true,
          class value_type                                = typename std::decay_t<L>::value_type,
          _has_binary_op_multiplies_enable_if<value_type> = true>
auto elementwise_product(L&& left, R&& right) {
    return apply_binary_op(std::forward<L>(left), std::forward<R>(right), std::multiplies<value_type>());
}

// --- Augmented assignment operator API ---
// -----------------------------------------

// These can just reuse corresponding binary operators while 'std::move()'ing lhs to avoid copying.

template <class L, class R, _are_tensors_with_same_value_type_enable_if<L, R> = true,
          class value_type = typename std::decay_t<L>::value_type, _has_assignment_op_plus_enable_if<value_type> = true>
L& operator+=(L&& left, R&& right) {
    return (left = std::move(left) + right);
}

template <class L, class R, _are_tensors_with_same_value_type_enable_if<L, R> = true,
          class value_type                               = typename std::decay_t<L>::value_type,
          _has_assignment_op_minus_enable_if<value_type> = true>
L& operator-=(L&& left, R&& right) {
    return (left = std::move(left) - right);
}

// --- Matrix multiplication ---
// -----------------------------

// Just like with binary operators, we run into the need to have 4 different implementations:
//    - (1)  dense +  dense =>  dense      (complexity O(N^3))
//    - (2)  dense + sparse =>  dense      (complexity O(N^2))
//    - (3) sparse +  dense =>  dense      (complexity O(N^2))
//    - (4) sparse + sparse => sparse      (complexity O(N)  )
//
// Would be gread to implement proper "smart" GEMM and CRS-multiplication for sparse matrices, however that adds
// an absolutely huge amount of complexity for x1.5-x3 speedup on matrix multiplication, which is already kind of
// an afterthought provided here mainly for convenience and not for actual number crunching usage.
//
// We do however use some basic optimizations like loop reordering / temporaries / blocking to have a sensible baseline,
// we properly account for sparsity which brings multiplication with sparse matrices from O(N^3) down to O(N^2) / O(N).

// (1)  dense +  dense =>  dense
//
// From benchmarks 1D blocking over "k" with a decently large block size seems to be more reliable than 2D/3D blocking.
// When measuring varying matrix sizes and datatypes, for matrices with less than ~1k rows/cols (aka matrices that fully
// fit into L2 cache) 1D blocking doesn't  seem to have much of an effect (positive or negative), while 2D/3D has often
// lead to a slowdown for specific matrix  sizes. At large enough sizes blocking leads to a noticeable (~2x) speedup.
// Note that this is all very hardware-specific, so it's difficult to make a truly generic judgement. In general,
// blocking seems to be worth it.
//
// Note that unlike other binary operators, here there is no possible benefit in r-value reuse.
//
template <class L, class R,                                                                                //
          _are_tensors_with_same_value_type_enable_if<L, R> = true,                                        //
          _is_nonsparse_tensor_enable_if<L>                 = true,                                        //
          _is_nonsparse_tensor_enable_if<R>                 = true,                                        //
          class value_type                                  = typename std::decay_t<L>::value_type,        //
          class return_type                                 = typename std::decay_t<L>::owning_reflection, //
          _has_binary_op_multiplies_enable_if<value_type>   = true,                                        //
          _has_assignment_op_plus_enable_if<value_type>     = true                                         //
          >
return_type operator*(const L& left, const R& right) {
    utl_mvl_assert(left.cols() == right.rows());

    using size_type = typename std::decay_t<L>::size_type;

    const size_type N_i = left.rows(), N_k = left.cols(), N_j = right.cols();
    // (N_i)x(N_k) * (N_k)x(N_j) => (N_i)x(N_j)

    constexpr size_type block_size_kk = 32;

    return_type res(N_i, N_j, value_type{});

    for (size_type kk = 0; kk < N_k; kk += block_size_kk) {
        const size_type k_extent = std::min(N_k, kk + block_size_kk);
        // needed for matrices that aren't a multiple of block size
        for (size_type i = 0; i < N_i; ++i) {
            for (size_type k = kk; k < k_extent; ++k) {
                const auto& r = left(i, k);
                for (size_type j = 0; j < N_j; ++j) res(i, j) += r * right(k, j);
            }
        }
    }

    return res;
}

// (2)  dense + sparse =>  dense

// TODO:

// (3) sparse +  dense =>  dense

// TODO:

// (4) sparse + sparse => sparse

// TODO:

// Clear out internal macros
#undef utl_mvl_tensor_arg_defs
#undef utl_mvl_tensor_arg_vals
#undef utl_mvl_require
#undef utl_mvl_reqs

} // namespace utl::mvl

#endif
#endif // module utl::mvl






// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DmitriBogdanov/prototyping_utils ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Module:        utl::parallel
// Documentation: https://github.com/DmitriBogdanov/prototyping_utils/blob/master/docs/module_parallel.md
// Source repo:   https://github.com/DmitriBogdanov/prototyping_utils
//
// This project is licensed under the MIT License
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#if !defined(UTL_PICK_MODULES) || defined(UTLMODULE_PARALLEL)
#ifndef UTLHEADERGUARD_PARALLEL
#define UTLHEADERGUARD_PARALLEL

// _______________________ INCLUDES _______________________

#include <condition_variable> // condition_variable
#include <cstddef>            // size_t
#include <functional>         // bind()
#include <future>             // future<>, packaged_task<>
#include <mutex>              // mutex, recursive_mutex, lock_guard<>, unique_lock<>
#include <queue>              // queue<>
#include <thread>             // thread
#include <type_traits>        // decay_t<>, invoke_result_t<>
#include <utility>            // forward<>()
#include <vector>             // vector

// ____________________ DEVELOPER DOCS ____________________

// In C++20 'std::jthread' can be used to simplify code a bit, no reason not to do so.
//
// In C++20 '_unroll<>()' template can be improved to take index as a template-lambda-explicit-argument
// rather than a regular arg, ensuring its constexpr'ness. This may lead to a slight performance boost
// as truly manual unrolling seems to be slightly faster than automatic one.

// ____________________ IMPLEMENTATION ____________________

namespace utl::parallel {

// =============
// --- Utils ---
// =============

std::size_t max_thread_count() {
    const std::size_t detected_threads = std::thread::hardware_concurrency();
    return detected_threads ? detected_threads : 1;
    // 'hardware_concurrency()' returns '0' if it can't determine the number of threads,
    // in this case we reasonably assume there is a single thread available
}

// No reason to include the entirety of <algorithm> just for 2 one-liner functions,
// so we implement 'std::size_t' min/max here
constexpr std::size_t _min_size(std::size_t a, std::size_t b) noexcept { return (b < a) ? b : a; }
constexpr std::size_t _max_size(std::size_t a, std::size_t b) noexcept { return (b < a) ? a : b; }

// We REALLY don't want compiler to mess up inlining of binary operations or unrolling template
// since that alone can tank the performance so we force inlining just to be sure
#if defined(_MSC_VER) || defined(_MSC_FULL_VER)
#define utl_parallel_force_inline __forceinline
#elif defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
#define utl_parallel_force_inline __attribute__((always_inline))
#else
#define utl_parallel_force_inline
#endif

// Template for automatic loop unrolling.
//
// It is used in 'parallel::reduce()' to speed up a tight loop while
// leaving the used with ability to easily control that unrolling.
//
// Benchmarks indicate speedups ~130% to ~400% depending on CPU, compiler and options
// large unrolling (32) seems to be the best on benchmarks, however it may bloat the binary
// and take over too many branch predictor slots in case of min/max reductions, 4-8 seems
// like a reasonable sweetspot for most machines. We use 4 as a less intrusive option.
//
// One may think that it is a job of compiler to perform such optimizations, yet even with
// GCC '-Ofast -funroll-all-loop' and GCC unroll pragmas it fails to do them reliably.
//
// The reason it fails to do so is pretty clear for '-O2' and below - strictly speaking, most binary
// operations on floats are non-commutative (sum depends on the order of addition, for example), however
// since reduction is inherently not-order-preserving there is no harm in reordering operations some more
// and unrolling the loop so compiler will be able to use SIMD if sees it as possile (it often does).
//
// Why unrolling still fails with '-Ofast' which reduce conformance and allows reordering
// of math operations is unclear, but it is how it happens when actually measured.
//
template <class T, T... indeces, class F>
utl_parallel_force_inline constexpr void _unroll_impl(std::integer_sequence<T, indeces...>, F&& f) {
    (f(std::integral_constant<T, indeces>{}), ...);
}
template <class T, T count, class F>
utl_parallel_force_inline constexpr void _unroll(F&& f) {
    _unroll_impl(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
}

// ===================
// --- Thread pool ---
// ===================

class ThreadPool {
private:
    std::vector<std::thread>     threads;
    mutable std::recursive_mutex thread_mutex;

    std::queue<std::packaged_task<void()>> tasks{};
    mutable std::mutex                     task_mutex;

    std::condition_variable task_cv;          // used to notify changes to the task queue
    std::condition_variable task_finished_cv; // used to notify of finished tasks

    // Signals
    bool stopping = false; // signal for workers to shut down '.worker_main()'
    bool paused   = false; // signal for workers to not pull new tasks from the queue
    bool waiting  = false; // signal for workers that they should notify 'task_finished_cv' when
                           // finishing a task, which is used to implement 'wait for tasks' methods

    int tasks_running = 0; // number of tasks currently executed by workers

    // Main function for worker threads,
    // here workers wait for the queue, pull new tasks from it and run them
    void thread_main() {
        bool task_was_finished = false;

        while (true) {
            std::unique_lock<std::mutex> task_lock(this->task_mutex);

            if (task_was_finished) {
                --this->tasks_running;
                if (this->waiting) this->task_finished_cv.notify_all();
                // no need to set 'task_was_finished' back to 'false',
                // the only way we get back into this condition is if another task was finished
            }

            // Pool isn't destructing, isn't paused and there are tasks available in the queue
            //    => continue execution, a new task from the queue and start executing it
            // otherwise
            //    => unlock the mutex and wait until a new task is submitted,
            //       pool is unpaused or destruction is initiated
            this->task_cv.wait(task_lock, [&] { return this->stopping || (!this->paused && !this->tasks.empty()); });

            if (this->stopping) break; // escape hatch for thread destruction

            // Pull a new task from the queue and start executing it
            std::packaged_task<void()> task_to_execute = std::move(this->tasks.front());
            this->tasks.pop();
            ++this->tasks_running;
            task_lock.unlock();

            task_to_execute(); // NOTE: Should I catch exceptions here?
            task_was_finished = true;
        }
    }

    void start_threads(std::size_t worker_count_increase) {
        const std::lock_guard<std::recursive_mutex> thread_lock(this->thread_mutex);
        // the mutex has to be recursive because we call '.start_threads()' inside '.set_num_threads()'
        // which also locks 'worker_mutex', if mutex wan't recursive we would deadlock trying to lock
        // it a 2nd time on the same thread.

        // NOTE: It feels like '.start_threads()' can be split into '.start_threads()' and
        // '._start_threads_assuming_locked()' which would remove the need for recursive mutex

        for (std::size_t i = 0; i < worker_count_increase; ++i)
            this->threads.emplace_back(&ThreadPool::thread_main, this);
    }

    void stop_all_threads() {
        const std::lock_guard<std::recursive_mutex> thread_lock(this->thread_mutex);

        {
            const std::lock_guard<std::mutex> task_lock(this->task_mutex);
            this->stopping = true;
            this->task_cv.notify_all();
        } // signals to all threads that they should stop running

        for (auto& worker : this->threads)
            if (worker.joinable()) worker.join();
        // 'joinable()' checks in needed so we don't try to join the master thread

        this->threads.clear();
    }

public:
    // --- Construction ---
    // --------------------

    ThreadPool() = default;

    explicit ThreadPool(std::size_t thread_count) { this->start_threads(thread_count); }

    ~ThreadPool() {
        this->unpause();
        this->wait_for_tasks();
        this->stop_all_threads();
    }

    // --- Threads ---
    // ---------------

    std::size_t get_thread_count() const {
        const std::lock_guard<std::recursive_mutex> thread_lock(this->thread_mutex);
        return this->threads.size();
    }

    void set_thread_count(std::size_t thread_count) {
        const std::size_t current_thread_count = this->get_thread_count();

        if (thread_count == current_thread_count) return;
        // 'quick escape' so we don't experience too much slowdown when the user calls '.set_thread_count()' reapeatedly

        if (thread_count > current_thread_count) {
            this->start_threads(thread_count - current_thread_count);
        } else {
            this->stop_all_threads();
            {
                const std::lock_guard<std::mutex> task_lock(this->task_mutex);
                this->stopping = false;
            }
            this->start_threads(thread_count);
            // It is possible to improve implementation by making the pool shrink by joining only the necessary amount
            // of threads instead of recreating the the whole pool, however that task is non-trivial and would likely
            // require a more granular signaling with one flag per thread instead of a global 'stopping' flag.
        }
    }

    // --- Task queue ---
    // ------------------

    template <class Func, class... Args>
    void add_task(Func&& func, Args&&... args) {
        const std::lock_guard<std::mutex> task_lock(this->task_mutex);
        this->tasks.emplace(std::bind(std::forward<Func>(func), std::forward<Args>(args)...));
        this->task_cv.notify_one(); // wakes up one thread (if possible) so it can pull the new task
    }

    template <class Func, class... Args,
              class FuncReturnType = std::invoke_result_t<std::decay_t<Func>, std::decay_t<Args>...>>
    [[nodiscard]] std::future<FuncReturnType> add_task_with_future(Func&& func, Args&&... args) {
#if defined(_MSC_VER) || defined(_MSC_FULL_VER)
        // MSVC messed up implementation of 'std::packaged_task<>' so it is not movable (which,
        // according to the standard, it should be) and they can't fix it for 7 (and counting) years
        // because fixing the bug would change the ABI. See this thread about the bug report:
        // https://developercommunity.visualstudio.com/t/unable-to-move-stdpackaged-task-into-any-stl-conta/108672
        // As a workaround we wrap the packaged task into a shared pointer and add another layer of packaging.
        auto new_task = std::make_shared<std::packaged_task<FuncReturnType()>>(
            std::bind(std::forward<Func>(func), std::forward<Args>(args)...));
        this->add_task([new_task] { (*new_task)(); }); // horrible
        return new_task->get_future();
#else
        std::packaged_task<FuncReturnType()> new_task(std::bind(std::forward<Func>(func), std::forward<Args>(args)...));
        auto                                 future = new_task.get_future();
        this->add_task(std::move(new_task));
        return future;
#endif
    }

    void wait_for_tasks() {
        std::unique_lock<std::mutex> task_lock(this->task_mutex);
        this->waiting = true;
        this->task_finished_cv.wait(task_lock, [&] { return this->tasks.empty() && this->tasks_running == 0; });
        this->waiting = false;
    }

    void clear_task_queue() {
        const std::lock_guard<std::mutex> task_lock(this->task_mutex);
        this->tasks = {}; // for some reason 'std::queue' has no '.clear()', complexity O(N)
    }

    // --- Pausing ---
    // ---------------

    void pause() {
        const std::lock_guard<std::mutex> task_lock(this->task_mutex);
        this->paused = true;
    }

    void unpause() {
        const std::lock_guard<std::mutex> task_lock(this->task_mutex);
        this->paused = false;
        this->task_cv.notify_all();
    }

    [[nodiscard]] bool is_paused() const {
        const std::lock_guard<std::mutex> task_lock(this->task_mutex);
        return this->paused;
    }
};

// =====================================
// --- Static thread pool operations ---
// =====================================

ThreadPool& static_thread_pool() {
    static ThreadPool pool(max_thread_count());
    return pool;
}

std::size_t get_thread_count() { return static_thread_pool().get_thread_count(); }

void set_thread_count(std::size_t thread_count) { static_thread_pool().set_thread_count(thread_count); }

// ================
// --- Task API ---
// ================

template <class Func, class... Args>
void task(Func&& func, Args&&... args) {
    static_thread_pool().add_task(std::forward<Func>(func), std::forward<Args>(args)...);
}

template <class Func, class... Args>
auto task_with_future(Func&& func, Args&&... args)
    -> std::future<std::invoke_result_t<std::decay_t<Func>, std::decay_t<Args>...>> {
    return static_thread_pool().add_task_with_future(std::forward<Func>(func), std::forward<Args>(args)...);
}

void wait_for_tasks() { static_thread_pool().wait_for_tasks(); }

// =======================
// --- Parallel ranges ---
// =======================

constexpr std::size_t default_grains_per_thread = 4;
// by default we distribute 4 tasks per thread, this number is purely empirical.
// We don't want to split up work into too many tasks (like with 'grain_size = 1')
// yet we want it to be a bit more granular than doing 1 task per threads since
// that would be horrible if tasks are uneven.

// Note:
// In range constructors we intentionally allow some possibly narrowing conversions like 'it1 - it2' to 'size_t'
// for better compatibility with containers that can use large ints as their difference type

template <class Idx>
struct IndexRange {
    Idx         first;
    Idx         last;
    std::size_t grain_size;

    IndexRange() = delete;
    constexpr IndexRange(Idx first, Idx last, std::size_t grain_size)
        : first(first), last(last), grain_size(grain_size) {}
    IndexRange(Idx first, Idx last)
        : IndexRange(first, last, _max_size(1, (last - first) / (get_thread_count() * default_grains_per_thread))){};
};

template <class Iter>
struct Range {
    Iter        begin;
    Iter        end;
    std::size_t grain_size;

    Range() = delete;
    constexpr Range(Iter begin, Iter end, std::size_t grain_size) : begin(begin), end(end), grain_size(grain_size) {}
    Range(Iter begin, Iter end)
        : Range(begin, end, _max_size(1, (end - begin) / (get_thread_count() * default_grains_per_thread))) {}


    template <class Container>
    Range(const Container& container) : Range(container.begin(), container.end()) {}

    template <class Container>
    Range(Container& container) : Range(container.begin(), container.end()) {}
}; // requires 'Iter' to be random-access-iterator

// User-defined deduction guides
//
// By default, template constructors cannot deduce template argument 'Iter',
// however it is possible to define a custom deduction guide and achieve what we want
//
// See: https://en.cppreference.com/w/cpp/language/class_template_argument_deduction#User-defined_deduction_guides
template <class Container>
Range(const Container& container) -> Range<typename Container::const_iterator>;

template <class Container>
Range(Container& container) -> Range<typename Container::iterator>;

// ==========================
// --- 'Parallel for' API ---
// ==========================

template <class Idx, class Func>
void for_loop(IndexRange<Idx> range, Func&& func) {
    for (Idx i = range.first; i < range.last; i += range.grain_size)
        task(std::forward<Func>(func), i, _min_size(i + range.grain_size, range.last));

    wait_for_tasks();
}

template <class Iter, class Func>
void for_loop(Range<Iter> range, Func&& func) {
    for (Iter i = range.begin; i < range.end; i += range.grain_size)
        task(std::forward<Func>(func), i, i + _min_size(range.grain_size, range.end - i));

    wait_for_tasks();
}

template <class Container, class Func>
void for_loop(const Container& container, Func&& func) {
    for_loop(Range{container}, std::forward<Func>(func));
}

template <class Container, class Func>
void for_loop(Container& container, Func&& func) {
    for_loop(Range{container}, std::forward<Func>(func));
}
// couldn't figure out how to make it work perfect-forwared 'Container&&',
// for some reason it would always cause template deduction to fail

// =============================
// --- 'Parallel reduce' API ---
// =============================

constexpr std::size_t default_unroll = 1;

template <std::size_t unroll = default_unroll, class Iter, class BinaryOp, class T = typename Iter::value_type>
auto reduce(Range<Iter> range, BinaryOp&& op) -> T {

    std::mutex result_mutex;
    T          result = *range.begin;
    // we have to start from the 1st element and not 'T{}' because there is no guarantee
    // than doing so would be correct for some non-trivial 'T' and 'op'

    for_loop(Range<Iter>{range.begin + 1, range.end, range.grain_size}, [&](Iter low, Iter high) {
        const std::size_t range_size = high - low;

        // Execute unrolled loop if unrolling is enabled and the range is sufficiently large
        if constexpr (unroll > 1)
            if (range_size > unroll) {
                // (parallel section) Compute partial result (unrolled for SIMD)
                // Reduce unrollable part
                std::array<T, unroll> partial_results;
                _unroll<std::size_t, unroll>([&](std::size_t j) { partial_results[j] = *(low + j); });
                Iter it = low + unroll;
                for (; it < high - unroll; it += unroll)
                    _unroll<std::size_t, unroll>(
                        [&, it](std::size_t j) { partial_results[j] = op(partial_results[j], *(it + j)); });
                // Reduce remaining elements
                for (; it < high; ++it)
                    partial_results[0] = op(partial_results[0], *it);
                // Collect the result
                for (std::size_t i = 1; i < partial_results.size(); ++i)
                    partial_results[0] = op(partial_results[0], partial_results[i]);

                // (critical section) Add partial result to the global one
                const std::lock_guard<std::mutex> result_lock(result_mutex);
                result = op(result, partial_results[0]);

                return; // skip the non-unrolled version
            }

        // Fallback onto a regular reduction loop otherwise
        // (parallel section) Compute partial result
        T partial_result = *low;
        for (auto it = low + 1; it != high; ++it) partial_result = op(partial_result, *it);

        // (critical section) Add partial result to the global one
        const std::lock_guard<std::mutex> result_lock(result_mutex);
        result = op(result, partial_result);
    });

    // Note 1:
    // We could also collect results into an array of partial results and then reduce it on the
    // main thread at the end, but that leads to a much less clean implementation and doesn't
    // seem to be measurably faster.

    // Note 2:
    // 'if constexpr (unroll > 1)' ensures that unrolling logic will have no effect
    //  whatsoever on the non-unrolled version of the template, it will not even compile.

    return result;
}

template <std::size_t unroll = default_unroll, class Container, class BinaryOp>
auto reduce(Container& container, BinaryOp&& op) -> typename Container::value_type {
    return reduce<unroll>(Range{container}, std::forward<BinaryOp>(op));
}

template <std::size_t unroll = default_unroll, class Container, class BinaryOp>
auto reduce(const Container& container, BinaryOp&& op) -> typename Container::value_type {
    return reduce<unroll>(Range{container}, std::forward<BinaryOp>(op));
}

// --- Pre-defined binary ops ---
// ------------------------------

// Note:
// Defining binary operations as free-standing function has a HUGE effect on
// parallel::reduce performance, for example on 4 threads:
//    binary op is 'sum'        => speedup ~90%-180%
//    binary op is 'struct sum' => speedup ~370%
// This is caused by failed inlining and seems to be the reason why standard library implements
// 'std::plus' as a functor class and not a free-standing function. I spent 2 hours of my life
// and 4 rewrites of 'parallel::reduce()' on this. Force-inline just to be sure.

template <class T>
struct sum {
    utl_parallel_force_inline constexpr T operator()(const T& lhs, const T& rhs) const { return lhs + rhs; }
};


template <class T>
struct prod {
    utl_parallel_force_inline constexpr T operator()(const T& lhs, const T& rhs) const { return lhs * rhs; }
};

template <class T>
struct min {
    utl_parallel_force_inline constexpr T operator()(const T& lhs, const T& rhs) const {
        return (rhs < lhs) ? rhs : lhs;
    }
};

template <class T>
struct max {
    utl_parallel_force_inline constexpr T operator()(const T& lhs, const T& rhs) const {
        return (rhs < lhs) ? lhs : rhs;
    }
};

// Clean up codegen macros
#undef utl_parallel_force_inline

} // namespace utl::parallel

#endif
#endif // module utl::parallel






// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DmitriBogdanov/prototyping_utils ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Module:        utl::predef
// Documentation: https://github.com/DmitriBogdanov/prototyping_utils/blob/master/docs/module_predef.md
// Source repo:   https://github.com/DmitriBogdanov/prototyping_utils
//
// This project is licensed under the MIT License
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#if !defined(UTL_PICK_MODULES) || defined(UTLMODULE_PREDEF)
#ifndef UTLHEADERGUARD_PREDEF
#define UTLHEADERGUARD_PREDEF

// _______________________ INCLUDES _______________________

#include <algorithm>   // fill_n()
#include <cctype>      // isspace()
#include <cstdlib>     // exit()
#include <iostream>    // cerr
#include <iterator>    // ostreambuf_iterator<>
#include <ostream>     // endl
#include <sstream>     // istringstream
#include <string>      // string, getline()
#include <string_view> // string_view
#include <utility>     // declval<>()

// ____________________ DEVELOPER DOCS ____________________

// Macros that provide a nicer way of querying some platform-specific stuff such as:
// compiler, platform, architecture, compilation info and etc.
//
// Boost Predef (https://www.boost.org/doc/libs/1_55_0/libs/predef/doc/html/index.html) provides
// a more complete package when it comes to supporing some esoteric platforms & compilers,
// but has a rather (in my opinion) ugly API.
//
// In addition utl::predef also provides some miscellaneous macros for automatic codegen, such as:
//    UTL_PREDEF_VA_ARGS_COUNT(args...)
//    UTL_PREDEF_ENUM_WITH_STRING_CONVERSION(enum_name, enum_values...)
//    UTL_PREDEF_IS_FUNCTION_DEFINED() - a nightmare of implementation, but it works
// some implementations may be rather sketchy due to trying to achieve things that weren't really
// meant to be achieved, but at the end of the day everything is standard-compliant.

// ____________________ IMPLEMENTATION ____________________

namespace utl::predef {

// ================================
// --- Compiler Detection Macro ---
// ================================

#if defined(_MSC_VER) || defined(_MSC_FULL_VER)
#define UTL_PREDEF_COMPILER_IS_MSVC
#elif defined(__GNUC__) || defined(__GNUC_MINOR__) || defined(__GNUC_PATCHLEVEL__)
#define UTL_PREDEF_COMPILER_IS_GCC
#elif defined(__clang__) || defined(__clang_major__) || defined(__clang_minor__) || defined(__clang_patchlevel__)
#define UTL_PREDEF_COMPILER_IS_CLANG
#elif defined(__llvm__)
#define UTL_PREDEF_COMPILER_IS_LLVM
#elif defined(__INTEL_COMPILER) || defined(__ICL) || defined(__ICC) || defined(__ECC)
#define UTL_PREDEF_COMPILER_IS_ICC
#elif defined(__PGI) || defined(__PGIC__) || defined(__PGIC_MINOR__) || defined(__PGIC_PATCHLEVEL__)
#define UTL_PREDEF_COMPILER_IS_PGI
#elif defined(__IBMCPP__) || defined(__xlC__) || defined(__xlc__)
#define UTL_PREDEF_COMPILER_IS_IBMCPP
#elif defined(__NVCC__) || defined(__CUDACC__)
#define UTL_PREDEF_COMPILER_IS_NVCC
#else
#define UTL_PREDEF_COMPILER_IS_UNKNOWN
#endif

constexpr std::string_view compiler_name =
#if defined(UTL_PREDEF_COMPILER_IS_MSVC)
    "MSVC"
#elif defined(UTL_PREDEF_COMPILER_IS_GCC)
    "GCC"
#elif defined(UTL_PREDEF_COMPILER_IS_CLANG)
    "clang"
#elif defined(UTL_PREDEF_COMPILER_IS_LLVM)
    "LLVM"
#elif defined(UTL_PREDEF_COMPILER_IS_ICC)
    "ICC"
#elif defined(UTL_PREDEF_COMPILER_IS_PGI)
    "PGI"
#elif defined(UTL_PREDEF_COMPILER_IS_IBMCPP)
    "IBMCPP"
#elif defined(UTL_PREDEF_COMPILER_IS_NVCC)
    "NVCC"
#else
    "<unknown>"
#endif
    ;

constexpr std::string_view compiler_full_name =
#if defined(UTL_PREDEF_COMPILER_IS_MSVC)
    "Microsoft Visual C++ Compiler"
#elif defined(UTL_PREDEF_COMPILER_IS_GCC)
    "GNU C/C++ Compiler"
#elif defined(UTL_PREDEF_COMPILER_IS_CLANG)
    "Clang Compiler"
#elif defined(UTL_PREDEF_COMPILER_IS_LLVM)
    "LLVM Compiler"
#elif defined(UTL_PREDEF_COMPILER_IS_ICC)
    "Inter C/C++ Compiler"
#elif defined(UTL_PREDEF_COMPILER_IS_PGI)
    "Portland Group C/C++ Compiler"
#elif defined(UTL_PREDEF_COMPILER_IS_IBMCPP)
    "IBM XL C/C++ Compiler"
#elif defined(UTL_PREDEF_COMPILER_IS_NVCC)
    "Nvidia Cuda Compiler Driver"
#else
    "<unknown>"
#endif
    ;

// ================================
// --- Platform Detection Macro ---
// ================================

#if defined(_WIN64) // _WIN64 implies _WIN32 so it should be first
#define UTL_PREDEF_PLATFORM_IS_WINDOWS_X64
#elif defined(_WIN32)
#define UTL_PREDEF_PLATFORM_IS_WINDOWS_X32
#elif defined(__CYGWIN__) && !defined(_WIN32) // Cygwin POSIX under Microsoft Window
#define UTL_PREDEF_PLATFORM_IS_CYGWIN
#elif defined(__ANDROID__) // __ANDROID__ implies __linux__ so it should be first
#define UTL_PREDEF_PLATFORM_IS_ANDROID
#elif defined(linux) || defined(__linux__) || defined(__linux)
#define UTL_PREDEF_PLATFORM_IS_LINUX
#elif defined(unix) || defined(__unix__) || defined(__unix)
#define UTL_PREDEF_PLATFORM_IS_UNIX
#elif defined(__APPLE__) && defined(__MACH__)
#define UTL_PREDEF_PLATFORM_IS_MACOS
#else
#define UTL_PREDEF_PLATFORM_IS_UNKNOWN
#endif

constexpr std::string_view platform_name =
#if defined(UTL_PREDEF_PLATFORM_IS_WINDOWS_X64)
    "Windows64"
#elif defined(UTL_PREDEF_PLATFORM_IS_WINDOWS_X32)
    "Windows32"
#elif defined(UTL_PREDEF_PLATFORM_IS_CYGWIN)
    "Windows (CYGWIN)"
#elif defined(UTL_PREDEF_PLATFORM_IS_ANDROID)
    "Android"
#elif defined(UTL_PREDEF_PLATFORM_IS_LINUX)
    "Linux"
#elif defined(UTL_PREDEF_PLATFORM_IS_UNIX)
    "Unix-like OS"
#elif defined(UTL_PREDEF_PLATFORM_IS_MACOS)
    "MacOS" // Apple OSX and iOS (Darwin)
#else
    "<unknown>"
#endif
    ;


// ====================================
// --- Architecture Detection Macro ---
// ====================================

#if defined(__x86_64) || defined(__x86_64__) || defined(__amd64__) || defined(__amd64) || defined(_M_X64)
#define UTL_PREDEF_ARCHITECTURE_IS_X86_64
#elif defined(i386) || defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__) ||             \
    defined(__i386) || defined(_M_IX86) || defined(_X86_) || defined(__THW_INTEL__) || defined(__I86__) ||             \
    defined(__INTEL__) || defined(__I86__) || defined(_M_IX86) || defined(__i686__) || defined(__i586__) ||            \
    defined(__i486__) || defined(__i386__)
#define UTL_PREDEF_ARCHITECTURE_IS_X86_32
#elif defined(__arm__) || defined(__thumb__) || defined(__TARGET_ARCH_ARM) || defined(__TARGET_ARCH_THUMB) ||          \
    defined(__TARGET_ARCH_ARM) || defined(__TARGET_ARCH_THUMB)
#define UTL_PREDEF_ARCHITECTURE_IS_ARM
#else
UTL_PREDEF_ARCHITECTURE_IS_UNKNOWN
#endif

constexpr std::string_view architecture_name =
#if defined(UTL_PREDEF_ARCHITECTURE_IS_X86_64)
    "x86-64"
#elif defined(UTL_PREDEF_ARCHITECTURE_IS_X86_32)
    "x86-32"
#elif defined(UTL_PREDEF_ARCHITECTURE_IS_ARM)
    "ARM"
#else
    "<unknown>"
#endif
    ;

// =========================================
// --- Language Standard Detection Macro ---
// =========================================

#if defined(UTL_PREDEF_COMPILER_IS_MSVC)
#define UTL_PREDEF_CPP_VERSION _MSVC_LANG
#else
#define UTL_PREDEF_CPP_VERSION __cplusplus
#endif
// Note 1:
// MSVC '__cplusplus' is defined, but stuck at '199711L'. It uses '_MSVC_LANG' instead.
//
// Note 2:
// '__cplusplus' is defined by the standard, it's only Microsoft who think standards are for other people.
//
// Note 3:
// MSVC has a flag '/Zc:__cplusplus' that enables standard behaviour for '__cplusplus'

#if (UTL_PREDEF_CPP_VERSION >= 202302L)
#define UTL_PREDEF_STANDARD_IS_23_PLUS
#elif (UTL_PREDEF_CPP_VERSION >= 202002L)
#define UTL_PREDEF_STANDARD_IS_20_PLUS
#elif (UTL_PREDEF_CPP_VERSION >= 201703L)
#define UTL_PREDEF_STANDARD_IS_17_PLUS
#elif (UTL_PREDEF_CPP_VERSION >= 201402L)
#define UTL_PREDEF_STANDARD_IS_14_PLUS
#elif (UTL_PREDEF_CPP_VERSION >= 201103L)
#define UTL_PREDEF_STANDARD_IS_11_PLUS
#else // everything below C++11 has the same value of '199711L'
#define UTL_PREDEF_STANDARD_IS_UNKNOWN
#endif
// Note:
// There should be no feasible way to fall below the 'UTL_PREDEF_STANDARD_IS_17_PLUS' since this library itself
// requires C++17 to compile, but might as well have a complete implementation for future reference.

constexpr std::string_view standard_name =
#if defined(UTL_PREDEF_STANDARD_IS_23_PLUS)
    "C++23"
#elif defined(UTL_PREDEF_STANDARD_IS_20_PLUS)
    "C++20"
#elif defined(UTL_PREDEF_STANDARD_IS_17_PLUS)
    "C++17"
#elif defined(UTL_PREDEF_STANDARD_IS_14_PLUS)
    "C++14"
#elif defined(UTL_PREDEF_STANDARD_IS_11_PLUS)
    "C++11"
#else
    "<unknown>"
#endif
    ;

// ========================================
// --- Compilation Mode Detection Macro ---
// ========================================

#if defined(_DEBUG)
#define UTL_PREDEF_MODE_IS_DEBUG
#endif

constexpr bool debug =
#if defined(UTL_PREDEF_MODE_IS_DEBUG)
    true
#else
    false
#endif
    ;

// ===========================
// --- Optimization macros ---
// ===========================

#if defined(UTL_PREDEF_COMPILER_IS_MSVC)
#define UTL_PREDEF_FORCE_INLINE __forceinline
#elif defined(UTL_PREDEF_COMPILER_IS_GCC) || defined(UTL_PREDEF_COMPILER_IS_CLANG) ||                                  \
    defined(UTL_PREDEF_COMPILER_IS_CLANG)
#define UTL_PREDEF_FORCE_INLINE __attribute__((always_inline))
#else
#define UTL_PREDEF_FORCE_INLINE
#endif

#if defined(UTL_PREDEF_STANDARD_IS_23_PLUS)
#define UTL_PREDEF_ASSUME [[assume(__VA_ARGS__))]]
#elif defined(UTL_PREDEF_COMPILER_IS_MSVC)
__assume(__VA_ARGS__)
#elif defined(UTL_PREDEF_COMPILER_IS_CLANG)
__builtin_assume(__VA_ARGS__)
#else // no equivalent GCC built-in
#endif

[[noreturn]] void unreachable() {
#if defined(UTL_PREDEF_STANDARD_IS_23_PLUS)
    std::unreachable();
#elif defined(UTL_PREDEF_COMPILER_IS_MSVC)
    __assume(false);
#elif defined(UTL_PREDEF_COMPILER_IS_GCC) || defined(UTL_PREDEF_COMPILER_IS_CLANG)
    __builtin_unreachable();
#endif
}

// ===================
// --- Other Utils ---
// ===================

[[nodiscard]] std::string compilation_summary() {
    std::string buffer;

    buffer += "Compiler:          ";
    buffer += compiler_full_name;
    buffer += '\n';

    buffer += "Platform:          ";
    buffer += platform_name;
    buffer += '\n';

    buffer += "Architecture:      ";
    buffer += architecture_name;
    buffer += '\n';

    buffer += "Compiled in DEBUG: ";
    buffer += debug ? "true" : "false";
    buffer += '\n';

    buffer += "Compiled under OS: ";
    buffer += __STDC_HOSTED__ ? "true" : "false";
    buffer += '\n';

    buffer += "Compilation date:  ";
    buffer += __DATE__;
    buffer += ' ';
    buffer += __TIME__;
    buffer += '\n';

    return buffer;
}

// - Size of __VA_ARGS__ in variadic macros -
#define _utl_expand_va_args(x_) x_ // a fix for MSVC bug not expanding __VA_ARGS__ properly

#define _utl_va_args_count_impl(x01_, x02_, x03_, x04_, x05_, x06_, x07_, x08_, x09_, x10_, x11_, x12_, x13_, x14_,    \
                                x15_, x16_, x17_, x18_, x19_, x20_, x21_, x22_, x23_, x24_, x25_, x26_, x27_, x28_,    \
                                x29_, x30_, x31_, x32_, x33_, x34_, x35_, x36_, x37_, x38_, x39_, x40_, x41_, x42_,    \
                                x43_, x44_, x45_, x46_, x47_, x48_, x49_, N_, ...)                                     \
    N_

#define UTL_PREDEF_VA_ARGS_COUNT(...)                                                                                  \
    _utl_expand_va_args(_utl_va_args_count_impl(__VA_ARGS__, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36,   \
                                                35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19,    \
                                                18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))

// - Enum with string conversion -
[[nodiscard]] inline std::string _trim_enum_string(const std::string& str) {
    std::string::const_iterator left_it = str.begin();
    while (left_it != str.end() && std::isspace(*left_it)) ++left_it;

    std::string::const_reverse_iterator right_it = str.rbegin();
    while (right_it.base() != left_it && std::isspace(*right_it)) ++right_it;

    return std::string(left_it, right_it.base()); // return string with whitespaces trimmed at both sides
}

inline void _split_enum_args(const char* va_args, std::string* strings, int count) {
    std::istringstream ss(va_args);
    std::string        buffer;

    for (int i = 0; ss.good() && (i < count); ++i) {
        std::getline(ss, buffer, ',');
        strings[i] = _trim_enum_string(buffer);
    }
};

#define UTL_PREDEF_ENUM_WITH_STRING_CONVERSION(enum_name_, ...)                                                        \
    namespace enum_name_ {                                                                                             \
    enum enum_name_ { __VA_ARGS__, _count };                                                                           \
                                                                                                                       \
    inline std::string _strings[_count];                                                                               \
                                                                                                                       \
    inline std::string to_string(enum_name_ enum_val) {                                                                \
        if (_strings[0].empty()) { utl::predef::_split_enum_args(#__VA_ARGS__, _strings, _count); }                    \
        return _strings[enum_val];                                                                                     \
    }                                                                                                                  \
                                                                                                                       \
    inline enum_name_ from_string(const std::string& enum_str) {                                                       \
        if (_strings[0].empty()) { utl::predef::_split_enum_args(#__VA_ARGS__, _strings, _count); }                    \
        for (int i = 0; i < _count; ++i) {                                                                             \
            if (_strings[i] == enum_str) { return static_cast<enum_name_>(i); }                                        \
        }                                                                                                              \
        return _count;                                                                                                 \
    }                                                                                                                  \
    }
    // We declare namespace with enum inside to simulate enum-class while having '_strings' array
    // and 'to_string()', 'from_string()' methods bundled with it.
    //
    // To count number of enum elements we add fake '_count' value at the end, which ends up being enum size
    //
    // '_strings' is declared compile-time, but gets filled through lazy evaluation upon first
    // 'to_string()' or 'from_string()' call. To fill it we interpret #__VA_ARGS__ as a single string
    // with some comma-separated identifiers. Those identifiers get split by commas, trimmed from
    // whitespaces and added to '_strings'
    //
    // Upon further calls (enum -> string) conversion is done though taking '_strings[enum_val]',
    // while (string -> enum) conversion requires searching through '_strings' to find enum index

#define UTL_PREDEF_IS_FUNCTION_DEFINED(function_name_, return_type_, ...)                                              \
    template <typename ReturnType, typename... ArgTypes>                                                               \
    class _utl_is_function_defined_impl_##function_name_ {                                                             \
    private:                                                                                                           \
        typedef char no[sizeof(ReturnType) + 1];                                                                       \
                                                                                                                       \
        template <typename... C>                                                                                       \
        static auto test(C... arg) -> decltype(function_name_(arg...));                                                \
                                                                                                                       \
        template <typename... C>                                                                                       \
        static no& test(...);                                                                                          \
                                                                                                                       \
    public:                                                                                                            \
        enum { value = (sizeof(test<ArgTypes...>(std::declval<ArgTypes>()...)) == sizeof(ReturnType)) };               \
    };                                                                                                                 \
                                                                                                                       \
    using is_function_defined_##function_name_ =                                                                       \
        _utl_is_function_defined_impl_##function_name_<return_type_, __VA_ARGS__>;
// TASK:
// We need to detect at compile time if function FUNC(ARGS...) exists.
// FUNC identifier isn't guaranteed to be declared.
//
// Ideal method would look like UTL_FUNC_EXISTS(FUNC, RETURN_TYPE, ARG_TYPES...) -> true/false
// This does not seem to be possible, we have to declare integral constant instead, see explanation below.
//
// WHY IS IT SO HARD:
// (1) Can this be done through preprocessor macros?
// No, preprocessor has no way to tell whether C++ identifier is defined or not.
//
// (2) Is there a compiler-specific way to do it?
// Doesn't seem to be the case.
//
// (3) Why not use some sort of template with FUNC as a parameter?
// Essentially we have to evaluate undeclared identifier, while compiler exits with error upon
// encountering anything undeclared. The only way to detect whether undeclared identifier exists
// or not seems to be through SFINAE.
//
// IMPLEMENTATION COMMENTS:
// We declate integral constant class with 2 functions 'test()', first one takes priority during overload
// resolution and compiles if FUNC(ARGS...) is defined, otherwise it's {Substitution Failure} which is
// {Is Not An Error} and second function compiles.
//
// To resolve which overload of 'test()' was selected we check the sizeof() return type, 2nd overload
// has a return type 'char[sizeof(ReturnType) + 1]' so it's always different from 1st overload.
// Resolution result (true/false) gets stored to '::value'.
//
// Note that we can't pass 'ReturnType' and 'ArgTypes' directly through '__VA_ARGS__' because
// to call function 'test(ARGS...)' in general case we have to 'std::declval<>()' all 'ARGS...'.
// To do so we can use variadic template syntax and then just forward '__VA_ARGS__' to the template
// through 'using is_function_present = is_function_present_impl<ReturnType, __VA_ARGS__>'.
//
// ALTERNATIVES: Perhaps some sort of tricky inline SFINAE can be done through C++14 generic lambdas.
//
// NOTE 1: Some versions of 'clangd' give a 'bugprone-sizeof-expression' warning for sizeof(*A),
// this is a false alarm.
//
// NOTE 2: Frankly, the usefullness of this is rather dubious since constructs like
//     if constexpr (is_function_defined_windown_specific) { <call the windows-specific function> }
//     else { <call the linux-specific function> }
// are still illegal due to 'if constexpr' requiting both branches to have defined identifiers,
// but since this arcane concept is already implemented why not keep it.

} // namespace utl::predef

#endif
#endif // module utl::predef






// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DmitriBogdanov/prototyping_utils ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Module:        utl::profiler
// Documentation: https://github.com/DmitriBogdanov/prototyping_utils/blob/master/docs/module_profiler.md
// Source repo:   https://github.com/DmitriBogdanov/prototyping_utils
//
// This project is licensed under the MIT License
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#if !defined(UTL_PICK_MODULES) || defined(UTLMODULE_PROFILER)
#ifndef UTLHEADERGUARD_PROFILER
#define UTLHEADERGUARD_PROFILER

// _______________________ INCLUDES _______________________

#include <algorithm>   // sort()
#include <chrono>      // chrono::steady_clock, chrono::duration_cast<>, std::chrono::milliseconds
#include <cstdlib>     // atexit()
#include <fstream>     // ofstream
#include <iomanip>     // setprecision(), setw()
#include <ios>         // streamsize, fixed,
#include <iostream>    // cout
#include <ostream>     // ostream
#include <sstream>     // ostringstream
#include <string>      // string
#include <string_view> // string_view
#include <vector>      // vector<>

// ____________________ DEVELOPER DOCS ____________________

// Macros for quick code profiling.
// Trivially simple, yet effective way of finding bottlenecks without any tooling.
//
// Can also be used to benchmark stuff "quick & dirty" due to doing all the time measuring
// and table formatting one would usually implement in their benchmarks. A proper bechmark
// suite of course would include support for automatic reruns and gather statistical data,
// but in prototying this is often not necessary.
//
// Resolving time recording inside recursion took some thinking, but ended up being quite simple in
// implementation. See the docs for more details on that.
//
// Currently, the overhead of profiling is barely different to the overhead of just time measurement,
// this also took a bit of thinking but in the end there is a nice solution that uses static variables
// to offload things that can only be done once to their initialization, and then links local variables
// to 'static' markers of the callsite recording. See 'UTL_PROFILE' macro for some more details.

// ____________________ IMPLEMENTATION ____________________

namespace utl::profiler {

// ==========================
// --- Profiler Internals ---
// ==========================

inline std::string _format_call_site(std::string_view file, int line, std::string_view func) {
    const std::string_view filename = file.substr(file.find_last_of("/\\") + 1);

    return (std::ostringstream() << filename << ":" << line << ", " << func << "()").str();
}

#define UTL_PROFILER_OPTION_USE_x86_INTRINSICS_FOR_FREQUENCY 3'300'000'000

#if !defined(UTL_PROFILER_OPTION_USE_x86_INTRINSICS_FOR_FREQUENCY)
using clock = std::chrono::steady_clock;
#else
struct clock {
    using rep                   = unsigned long long int;
    using period                = std::ratio<1, UTL_PROFILER_OPTION_USE_x86_INTRINSICS_FOR_FREQUENCY>;
    using duration              = std::chrono::duration<rep, period>;
    using time_point            = std::chrono::time_point<clock>;
    static const bool is_steady = true;

    static time_point now() noexcept {
        unsigned int low, high;
        asm volatile("rdtsc" : "=a"(low), "=d"(high)); // GCC/clang asm intrinsic, MSVC uses __asm() with more overhead
        return time_point(duration(static_cast<rep>(high) << 32 | low));
    }
};
#endif

using duration   = clock::duration;
using time_point = clock::time_point;

inline const time_point _program_entry_time_point = clock::now();

struct _record {
    const char* file;
    int         line;
    const char* func;
    const char* label;
    duration    accumulated_time;
};

inline void _utl_profiler_atexit(); // predeclaration, implementation has circular dependency with 'RecordManager'

// =========================
// --- Profiler Classess ---
// =========================

class _record_manager {
private:
    _record data;

public:
    inline static std::vector<_record> records;
    inline static int                  exclusive_recursion{};
    int                                recursion{};

    void add_time(duration time) noexcept { this->data.accumulated_time += time; }

    _record_manager() = delete;

    _record_manager(const char* file, int line, const char* func, const char* label)
        : data({file, line, func, label, duration(0)}) {
        // 'file', 'func', 'label' are guaranteed to be string literals, since we want to
        // have as little overhead as possible during runtime, we can just save raw pointers
        // and convert them to nicer types like 'std::string_view' later in the formatting stage

        // Profiler ever gets called => register result output at 'std::exit()'
        static bool first_call = true;
        if (first_call) {
            std::atexit(_utl_profiler_atexit);
            first_call = false;
        }
    }

    ~_record_manager() { records.emplace_back(this->data); }
};

// We need 4 slightly different timer classes, so might as well deduplicate some code by moving it into a base class
struct _timer_base {
protected:
    time_point       start;
    _record_manager* record_manager;
    // we could use 'std::optional<std::reference_wrapper<RecordManager>>',
    // but that would inctroduce more dependencies for no real reason
public:
    constexpr operator bool() const noexcept { return true; }

    _timer_base(_record_manager* manager) : record_manager(manager) {}
};

// Simple class that records the time of its creation and destruction and records it into the connected 'RecordManager'
struct _scope_timer : public _timer_base {
    _scope_timer(_record_manager* manager) : _timer_base(manager) {
        if (this->record_manager->recursion++ == 0) this->start = clock::now();
        // this check prevent timer from double-counting time spent inside
        // of it's own scope due to recursive calls
    }

    ~_scope_timer() {
        if (--this->record_manager->recursion == 0) this->record_manager->add_time(clock::now() - this->start);
    }
};

// Same thing as '_scope_timer' except it uses global static 'exclusive_recursion' instead of regular 'recursion' that
// is specific to each '_record_manager'. This effecively means no '_exclusive_scope_timer''s will count time as long a
// single instance of another exclusive timer exists. This allows us to resolve som tricky situations such as recursion
struct _exclusive_scope_timer : public _timer_base {
    _exclusive_scope_timer(_record_manager* manager) : _timer_base(manager) {
        if (this->record_manager->exclusive_recursion++ == 0) this->start = clock::now();
    }

    ~_exclusive_scope_timer() {
        if (--this->record_manager->exclusive_recursion == 0)
            this->record_manager->add_time(clock::now() - this->start);
    }
};

// Same thing as '_scope_timer', except instead of destructor it uses an explicitly called method to record time.
// We need it to implement code-segment profiling with 'UTL_PROFILER_BEGIN' and 'UTL_PROFILER_END'
struct _segment_timer : public _timer_base {
    _segment_timer(_record_manager* manager) : _timer_base(manager) {
        if (this->record_manager->recursion++ == 0) this->start = clock::now();
    }

    void finish() {
        if (--this->record_manager->recursion == 0) this->record_manager->add_time(clock::now() - this->start);
    }
};

struct _exclusive_segment_timer : public _timer_base {
    _exclusive_segment_timer(_record_manager* manager) : _timer_base(manager) {
        if (this->record_manager->exclusive_recursion++ == 0) this->start = clock::now();
    }

    void finish() {
        if (--this->record_manager->exclusive_recursion == 0)
            this->record_manager->add_time(clock::now() - this->start);
    }
};

// ==================================
// --- Profiler Exit & Formatting ---
// ==================================

inline void _utl_profiler_atexit() {
    // NOTE:
    // Lots of ugly formatting stuff here, but it works, so a nicer rewrite is low-priority.
    // Would make a lot more sense to format all the stuff into a `matrix` of strings first,
    // and then do all the sorting/column width adjustment and etc. Should be faster (which
    // doesn't really matter here) and more concise too.

    const auto total_runtime = clock::now() - _program_entry_time_point;

    std::ostream* os = &std::cout;

    // Convenience functions
    const auto duration_to_sec = [](duration duration) -> double {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count() / 1e9;
    };

    const auto duration_percentage = [&](double duration_sec) -> double {
        const double total_runtime_sec = duration_to_sec(total_runtime);
        return (duration_sec / total_runtime_sec) * 100.;
    };

    const auto float_printed_size = [](double value, std::streamsize precision, decltype(std::fixed) format,
                                       std::string_view postfix) -> std::streamsize {
        std::ostringstream ss;
        ss << std::setprecision(precision) << format << value << postfix;
        return ss.str().size(); // can be done faster but we don't really care here
    };

    const auto repeat_hline_symbol = [](std::streamsize repeats) -> std::string {
        return std::string(static_cast<size_t>(repeats), '-');
    };

    constexpr std::streamsize  duration_precision = 2;
    constexpr auto             duration_format    = std::fixed;
    constexpr std::string_view duration_postfix   = " s";

    constexpr std::streamsize  percentage_precision = 1;
    constexpr auto             percentage_format    = std::fixed;
    constexpr std::string_view percentage_postfix   = "%";

    // Sort records by their accumulated time
    std::sort(_record_manager::records.begin(), _record_manager::records.end(),
              [](const _record& l, const _record& r) { return l.accumulated_time > r.accumulated_time; });

    // Collect max length of each column (for proper formatting)
    constexpr std::string_view column_name_call_site  = "Call Site";
    constexpr std::string_view column_name_label      = "Label";
    constexpr std::string_view column_name_duration   = "Time";
    constexpr std::string_view column_name_percentage = "Time %";

    std::streamsize max_length_call_site  = column_name_call_site.size();
    std::streamsize max_length_label      = column_name_label.size();
    std::streamsize max_length_duration   = column_name_duration.size();
    std::streamsize max_length_percentage = column_name_percentage.size();

    for (const auto& record : _record_manager::records) {
        const std::string call_site    = _format_call_site(record.file, record.line, record.func);
        const std::string label        = record.label;
        const double      duration_sec = duration_to_sec(record.accumulated_time);

        // 'Call Site' column
        const std::streamsize length_call_site = call_site.size();
        if (max_length_call_site < length_call_site) max_length_call_site = length_call_site;

        // 'Label' column
        const std::streamsize length_label = label.size();
        if (max_length_label < length_label) max_length_label = length_label;

        // 'Time' column
        const std::streamsize length_duration =
            float_printed_size(duration_sec, duration_precision, duration_format, duration_postfix);
        if (max_length_duration < length_duration) max_length_duration = length_duration;

        // 'Time %' column
        const auto            percentage = duration_percentage(duration_sec);
        const std::streamsize length_percentage =
            float_printed_size(percentage, percentage_precision, percentage_format, percentage_postfix);
        if (max_length_percentage < length_percentage) max_length_percentage = length_percentage;
    }

    // Print formatted profiler header
    constexpr std::string_view HEADER_TEXT = " UTL PROFILING RESULTS ";

    const std::streamsize total_table_length = sizeof("| ") - 1 + max_length_call_site + sizeof(" | ") - 1 +
                                               max_length_label + sizeof(" | ") - 1 + max_length_duration +
                                               sizeof(" | ") - 1 + max_length_percentage + sizeof(" |") -
                                               1; // -1 because sizeof(char[]) accounts for invisible '\0' at the end

    const std::streamsize header_text_length = HEADER_TEXT.size();
    const std::streamsize header_left_pad    = (total_table_length - header_text_length) / 2;
    const std::streamsize header_right_pad   = total_table_length - header_text_length - header_left_pad;

    *os << "\n"
        << repeat_hline_symbol(header_left_pad + 1) << HEADER_TEXT << repeat_hline_symbol(header_right_pad + 1)
        << '\n'
        // + 1 makes header hline extend 1 character past the table on both sides
        << "\n"
        << " Total runtime -> " << std::setprecision(duration_precision) << duration_format
        << duration_to_sec(total_runtime) << " sec\n"
        << "\n";

    // Print formatted table header
    *os << " | " << std::setw(max_length_call_site) << column_name_call_site << " | " << std::setw(max_length_label)
        << column_name_label << " | " << std::setw(max_length_duration) << column_name_duration << " | "
        << std::setw(max_length_percentage) << column_name_percentage << " |\n";

    *os << " |"
        << repeat_hline_symbol(max_length_call_site + 2) // add 2 to account for delimers not having spaces in hline
        << "|" << repeat_hline_symbol(max_length_label + 2) << "|" << repeat_hline_symbol(max_length_duration + 2)
        << "|" << repeat_hline_symbol(max_length_percentage + 2) << "|\n";

    *os << std::setfill(' '); // reset the fill so we don't mess with table contents


    // Print formatted table contents
    for (const auto& record : _record_manager::records) {
        const std::string call_site    = _format_call_site(record.file, record.line, record.func);
        const std::string label        = record.label;
        const double      duration_sec = duration_to_sec(record.accumulated_time);
        const double      percentage   = duration_percentage(duration_sec);

        // Join floats with their postfixes into a single string so they are properly handled by std::setw()
        // (which only affects the first value leading to a table misaligned by postfix size)
        std::ostringstream ss_duration;
        ss_duration << std::setprecision(duration_precision) << duration_format << duration_sec << duration_postfix;

        std::ostringstream ss_percentage;
        ss_percentage << std::setprecision(percentage_precision) << percentage_format << percentage
                      << percentage_postfix;

        *os << " | " << std::setw(max_length_call_site) << call_site << " | " << std::setw(max_length_label) << label
            << " | " << std::setw(max_length_duration) << ss_duration.str() << " | " << std::setw(max_length_percentage)
            << ss_percentage.str() << " |\n";
    }
}

// ========================
// --- Profiler Codegen ---
// ========================

#define _utl_profiler_concat_tokens(a, b) a##b
#define _utl_profiler_concat_tokens_wrapper(a, b) _utl_profiler_concat_tokens(a, b)
#define _utl_profiler_add_uuid(varname_) _utl_profiler_concat_tokens_wrapper(varname_, __LINE__)
// This macro creates token 'varname_##__LINE__' from 'varname_'.
//
// The reason we can't just write it as is, is that function-macros only expands their macro-arguments
// if neither the stringizing operator # nor the token-pasting operator ## are applied to the arguments
// inside the macro body.
//
// Which means in a simple 'varname_##__LINE__' macro '__LINE__' doesn't expand to it's value.
//
// We can get around this fact by introducing indirection,
// '__LINE__' gets expanded in '_utl_profiler_concat_tokens_wrapper()'
// and then tokenized and concatenated in '_utl_profiler_concat_tokens()'

// --- Scope profiling ---
// -----------------------

#define UTL_PROFILER(label_)                                                                                           \
    constexpr bool _utl_profiler_add_uuid(utl_profiler_macro_guard_) = true;                                           \
                                                                                                                       \
    static_assert(_utl_profiler_add_uuid(utl_profiler_macro_guard_), "UTL_PROFILE is a multi-line macro.");            \
                                                                                                                       \
    static utl::profiler::_record_manager _utl_profiler_add_uuid(utl_profiler_record_manager_)(__FILE__, __LINE__,     \
                                                                                               __func__, label_);      \
                                                                                                                       \
    if constexpr (const utl::profiler::_scope_timer _utl_profiler_add_uuid(utl_profiler_scope_timer_){                 \
                      &_utl_profiler_add_uuid(utl_profiler_record_manager_)})
// Note 1:
//
//    constexpr bool ... = true;
//    static_assert(..., "UTL_PROFILE is a multi-line macro.");
//
// is reponsible for preventing accidental errors caused by using macro like this:
//
//    for (...) UTL_PROFILER("") function(); // will only loop the first line of the multi-line macro
//
// If someone tries to write it like this, the constexpr bool variable will be "pulled" into a narrower scope,
// causing 'static_assert()' to fail due to using undeclared identifier. Since the line with undeclared identifier
// gets expanded, the user will be able to see the assert message.
//
// Note 2:
//
// By separating "record management" into a static variable and "actual timing" into a non-static one,
// we can avoid additional overhead from having to locate the record, corresponding to the profiled source location.
// (an operation that requires a non-trivial vector/map lookup with string comparisons)
//
// Static variable initializes its record once and timer does the bare minimum of work - 2 calls to 'now()' to get
// timing, one addition to accumulated time and a check for recursion (so it can skip time appropriately).
//
//  Note 3:
//
// _utl_profiler_add_uuid(...) ensures no identifier collisions when several profilers exist in a single scope.
// Since in this context 'uuid' is a line number, the only case in which ids can collide is when multiple profilers
// are declated on the same line, which I assume no sane person would do. And even if they would, that would simply
// lead to a compiler error. Can't really do better than that without resorting to non-standard macros like
// '__COUNTER__' for 'uuid' creation

#define UTL_PROFILER_EXCLUSIVE(label_)                                                                                 \
    constexpr bool _utl_profiler_add_uuid(utl_profiler_macro_guard_) = true;                                           \
                                                                                                                       \
    static_assert(_utl_profiler_add_uuid(utl_profiler_macro_guard_), "UTL_PROFILER_EXCLUSIVE is a multi-line macro."); \
                                                                                                                       \
    static utl::profiler::_record_manager _utl_profiler_add_uuid(utl_profiler_record_manager_)(__FILE__, __LINE__,     \
                                                                                               __func__, label_);      \
                                                                                                                       \
    if constexpr (const utl::profiler::_exclusive_scope_timer _utl_profiler_add_uuid(utl_profiler_scope_timer_){       \
                      &_utl_profiler_add_uuid(utl_profiler_record_manager_)})
// Note:
//
// Exact same thing as a regular UTL_PROFILER() but uses '_exclusive_scope_timer' instead.
// The reason we need this for recursion is nicely explained in the docs.

// --- Segment profiling ---
// -------------------------

#define UTL_PROFILER_BEGIN(segment_label_, label_)                                                                     \
    static utl::profiler::_record_manager utl_profiler_record_manager_##segment_label_(__FILE__, __LINE__, __func__,   \
                                                                                       label_);                        \
    utl::profiler::_segment_timer         utl_profiler_segment_timer_##segment_label_(                                 \
        &utl_profiler_record_manager_##segment_label_)

#define UTL_PROFILER_END(segment_label_) utl_profiler_segment_timer_##segment_label_.finish()
// Note 1:
//
// Last semicolon is intentiomally skipped so macro requires it at the end and
// doesn't mess up auto code formatters that have a dislike for statement macros.
//
// Note 2:
//
// The idea here exactly the same as with scope profiles, except instead of '_scope_timer' we use '_segment_timer'
// that records time on a '.finish()' call instead of destructor. We can put this call inside the END macro
// and have a nice 2-macro API for profiling segments without creating a scope.

#define UTL_PROFILER_EXCLUSIVE_BEGIN(segment_label_, label_)                                                           \
    static utl::profiler::_record_manager   utl_profiler_record_manager_##segment_label_(__FILE__, __LINE__, __func__, \
                                                                                         label_);                      \
    utl::profiler::_exclusive_segment_timer utl_profiler_segment_timer_##segment_label_(                               \
        &utl_profiler_record_manager_##segment_label_)

#define UTL_PROFILER_EXCLUSIVE_END(segment_label_) utl_profiler_segment_timer_##segment_label_.finish()

} // namespace utl::profiler

#endif
#endif // macro-module UTL_PROFILER






// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DmitriBogdanov/prototyping_utils ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Module:        utl::progressbar
// Documentation: https://github.com/DmitriBogdanov/prototyping_utils/blob/master/docs/module_progressbar.md
// Source repo:   https://github.com/DmitriBogdanov/prototyping_utils
//
// This project is licensed under the MIT License
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#if !defined(UTL_PICK_MODULES) || defined(UTLMODULE_PROGRESSBAR)
#ifndef UTLHEADERGUARD_PROGRESSBAR
#define UTLHEADERGUARD_PROGRESSBAR

// _______________________ INCLUDES _______________________

#include <algorithm> // fill_n
#include <chrono>    // chrono::steady_clock, chrono::time_point<>, chrono::duration_cast<>, chrono::seconds
#include <cmath>     // floor()
#include <cstddef>   // size_t
#include <iomanip>   // setprecision()
#include <ios>       // fixed
#include <iostream>  // cout
#include <iterator>  // ostreambuf_iterator<>
#include <ostream>   // ostream
#include <sstream>   // ostringstream
#include <string>    // string

// ____________________ DEVELOPER DOCS ____________________

// Simple progress bars for terminal applications. Rendered in ASCII on the main thread with manual updates
// for maximal compatibility. Perhaps can be extended with some fancier async options that display animations.
//
// # ::set_ostream() #
// Sets ostream used for progress bars.
//
// # ::Percentage #
// Proper progress bar, uses carriage return escape sequence (\r) to render new states in the same spot.
// Shows an estimate of remaining time.
//
// # ::Ruler #
// Primitive & lightweight progress bar, useful when terminal has no proper support for escape sequences.

// ____________________ IMPLEMENTATION ____________________

namespace utl::progressbar {

inline std::ostream* _output_stream = &std::cout;

inline void set_ostream(std::ostream& new_ostream) { _output_stream = &new_ostream; }

class Percentage {
private:
    char done_char;
    char not_done_char;
    bool show_time_estimate;

    std::size_t length_total;   // full   bar length
    std::size_t length_current; // filled bar length

    double last_update_percentage;
    double update_rate;

    using Clock     = std::chrono::steady_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    TimePoint timepoint_start; // used to estimate remaining time
    TimePoint timepoint_current;

    int previous_string_length; // used to properly return the carriage when dealing with changed string size

    void draw_progressbar(double percentage) {
        const auto displayed_percentage = this->update_rate * std::floor(percentage / this->update_rate);
        // floor percentage to a closest multiple of 'update_rate' for nicer display
        // since actual updates are only required to happen no more ofter than 'update_rate'
        // and don't have to correspond to exact multiples of it

        // Estimate remaining time (linearly) and format it min + sec
        const auto time_elapsed       = this->timepoint_current - this->timepoint_start;
        const auto estimate_full      = time_elapsed / percentage;
        const auto estimate_remaining = estimate_full - time_elapsed;

        const auto estimate_remaining_sec = std::chrono::duration_cast<std::chrono::seconds>(estimate_remaining);

        const auto displayed_min = (estimate_remaining_sec / 60ll).count();
        const auto displayed_sec = (estimate_remaining_sec % 60ll).count();

        const bool show_min  = (displayed_min != 0);
        const bool show_sec  = (displayed_sec != 0) && !show_min;
        const bool show_time = (estimate_remaining_sec.count() > 0);

        std::ostringstream ss;

        // Print bar
        ss << '[';
        std::fill_n(std::ostreambuf_iterator<char>(ss), this->length_current, this->done_char);
        std::fill_n(std::ostreambuf_iterator<char>(ss), this->length_total - this->length_current, this->not_done_char);
        ss << ']';

        // Print percentage
        ss << ' ' << std::fixed << std::setprecision(2) << 100. * displayed_percentage << '%';

        // Print time estimate
        if (this->show_time_estimate && show_time) {
            ss << " (remaining:";
            if (show_min) ss << ' ' << displayed_min << " min";
            if (show_sec) ss << ' ' << displayed_sec << " sec";
            ss << ')';
        }

        const std::string bar_string = ss.str();

        // Add spaces at the end to overwrite the previous string if it was longer that current
        const int current_string_length = static_cast<int>(bar_string.length());
        const int string_length_diff    = this->previous_string_length - current_string_length;

        if (string_length_diff > 0) { std::fill_n(std::ostreambuf_iterator<char>(ss), string_length_diff, ' '); }

        this->previous_string_length = current_string_length;

        // Return the carriage
        (*_output_stream) << ss.str(); // don't reuse 'bar_string', now 'ss' can also contain spaces at the end
        (*_output_stream) << '\r';
        (*_output_stream).flush();
        // '\r' returns cursor to the beginning of the line => most sensible consoles will
        // render render new lines over the last one. Otherwise every update produces a
        // bar on a new line, which looks worse but isn't critical for the purpose.
    }

public:
    Percentage(char done_char = '#', char not_done_char = '.', std::size_t bar_length = 30, double update_rate = 1e-2,
               bool show_time_estimate = true)
        : done_char(done_char), not_done_char(not_done_char), show_time_estimate(show_time_estimate),
          length_total(bar_length), length_current(0), last_update_percentage(0), update_rate(update_rate),
          timepoint_start(Clock::now()), previous_string_length(static_cast<int>(bar_length) + sizeof("[] 100.00%")) {}

    void start() {
        this->last_update_percentage = 0.;
        this->length_current         = 0;
        this->timepoint_start        = Clock::now();
        (*_output_stream) << '\n';
    }

    void set_progress(double percentage) {
        if (percentage - this->last_update_percentage <= this->update_rate) return;

        this->last_update_percentage = percentage;
        this->length_current         = static_cast<std::size_t>(percentage * static_cast<double>(this->length_total));
        this->timepoint_current      = Clock::now();
        this->draw_progressbar(percentage);
    }

    void finish() {
        this->last_update_percentage = 1.;
        this->length_current         = this->length_total;
        this->draw_progressbar(1.);
        (*_output_stream) << '\n';
    }
};

class Ruler {
private:
    char done_char;

    std::size_t length_total;
    std::size_t length_current;

public:
    Ruler(char done_char = '#') : done_char(done_char), length_total(51), length_current(0) {}

    void start() {
        this->length_current = 0;

        (*_output_stream) << '\n'
                          << " 0    10   20   30   40   50   60   70   80   90   100%\n"
                          << " |----|----|----|----|----|----|----|----|----|----|\n"
                          << ' ';
    }

    void set_progress(double percentage) {
        const std::size_t length_new = static_cast<std::size_t>(percentage * static_cast<double>(this->length_total));

        if (length_new > length_current) {
            const auto chars_to_add = length_new - this->length_current;
            std::fill_n(std::ostreambuf_iterator<char>(*_output_stream), chars_to_add, this->done_char);
        }

        this->length_current = length_new;
    }

    void finish() {
        if (this->length_total > this->length_current) {
            const auto chars_to_add = this->length_total - this->length_current;
            std::fill_n(std::ostreambuf_iterator<char>(*_output_stream), chars_to_add, this->done_char);
        }

        this->length_current = this->length_total;

        (*_output_stream) << '\n';
    }
};

} // namespace utl::progressbar

#endif
#endif // module utl::progressbar






// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DmitriBogdanov/prototyping_utils ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Module:        utl::random
// Documentation: https://github.com/DmitriBogdanov/prototyping_utils/blob/master/docs/module_random.md
// Source repo:   https://github.com/DmitriBogdanov/prototyping_utils
//
// This project is licensed under the MIT License
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#if !defined(UTL_PICK_MODULES) || defined(UTLMODULE_RANDOM)
#ifndef UTLHEADERGUARD_RANDOM
#define UTLHEADERGUARD_RANDOM

// _______________________ INCLUDES _______________________

#include <cstdint>          // uint64_t
#include <ctime>            // time()
#include <initializer_list> // initializer_list<>
#include <limits>           // numeric_limits<>::digits
#include <limits>           // numeric_limits<>::min(), numeric_limits<>::max()
#include <random>           // random_device, std::uniform_int_distribution<>,
                            // std::uniform_real_distribution<>, generate_canonical<>

// ____________________ DEVELOPER DOCS ____________________

// Implements a proper modern PRNG engine, compatible with std <random>.
// Adds 'sensible std <random> wrappers' for people who aren't fond of writing
// 3 lines code just to get a simple rand value.
//
// # XorShift64StarGenerator #
// Random 'std::uint64_t' generator that satisfies uniform random number generator requirements
// from '<random>' (see https://en.cppreference.com/w/cpp/named_req/UniformRandomBitGenerator).
// Implementation "XorShift64* suggested by Marsaglia G. in 2003 "Journal of Statistical Software"
// (see https://www.jstatsoft.org/article/view/v008i14).
// Slightly faster than 'std::rand()', while providing higher quality random.
// Significantly faster than 'std::mt19937'.
// State consists of a single 'std::uint64_t', requires seed >= 1.
//
// # xorshift64star #
// Global instance of XorShift64StarGenerator.
//
// # ::seed(), ::seed_with_time(), ::seed_with_random_device() #
// Seeds random with value/current_time/random_device.
// Random device is a better source of entropy, however it's more expensive to initialize
// than just taking current time with <ctime>, in some cases a "worse by lightweigh" can
// be prefered.
//
// # ::rand_int(), ::rand_uint(), ::rand_float(), ::rand_double() #
// Random value in [min, max] range.
// Floats with no parameters assume range [0, 1].
//
// # ::rand_bool() #
// Randomly chooses 0 or 1.
//
// # ::rand_choise() #
// Randomly chooses a value from initializer list.
//
// # ::rand_linear_combination() #
// Produces "c A + (1-c) B" with random "0 < c < 1" assuming objects "A", "B" support arithmetic operations.
// Useful for vector and color operations.

// ____________________ IMPLEMENTATION ____________________

namespace utl::random {

// =========================
// --- Random Generators ---
// =========================

class XorShift64StarGenerator {
    // meets uniform random number generator requirements
    // (see https://en.cppreference.com/w/cpp/named_req/UniformRandomBitGenerator)
    //
    // implementation XorShift64* suggested by Marsaglia G. in 2003 "Journal of Statistical Software"
    // (see https://www.jstatsoft.org/article/view/v008i14)
public:
    using result_type = std::uint64_t;

private:
    result_type state;

public:
    XorShift64StarGenerator(result_type seed = 0) : state(seed + 1) {}

    [[nodiscard]] static constexpr result_type min() { return std::numeric_limits<result_type>::min(); }
    [[nodiscard]] static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }

    void seed(result_type seed) { this->state = seed + 1; } // enforce seed >= 1 by always adding +1 to uint

    result_type operator()() {
        this->state ^= this->state >> 12;
        this->state ^= this->state << 25;
        this->state ^= this->state >> 27;
        return this->state * 0x2545F4914F6CDD1DULL;
    }
};

inline XorShift64StarGenerator xorshift64star;

// ===============
// --- Seeding ---
// ===============

inline void seed(XorShift64StarGenerator::result_type random_seed) { xorshift64star.seed(random_seed); }

inline void seed_with_time() { utl::random::seed(static_cast<XorShift64StarGenerator::result_type>(std::time(NULL))); }

inline void seed_with_random_device() {
    std::random_device rd;
    utl::random::seed(static_cast<XorShift64StarGenerator::result_type>(rd()));
}

// ========================
// --- Random Functions ---
// ========================

// Note 1:
// Despite the intuitive judgement, benchmarks don't seem to indicate that creating
// 'std::uniform_..._distribution<>' on each call introduces any noticeble overhead
//
// sizeof(std::uniform_int_distribution<int>) == 8
// sizeof(std::uniform_real_distribution<double>) == 16

// Note 2:
// No '[[nodiscard]]' since random functions inherently can't be pure due to advancing the generator state.
// Discarding return values while not very sensible, can still be done for the sake of advancing state.
// Ideally we would want users to advance the state directly, but I'm not sure how to communicate that in
// '[[nodiscard]]' warnings.

inline int rand_int(int min, int max) {
    std::uniform_int_distribution<int> distr{min, max};
    return distr(xorshift64star);
}

inline int rand_uint(unsigned int min, unsigned int max) {
    std::uniform_int_distribution<unsigned int> distr{min, max};
    return distr(xorshift64star);
}

inline float rand_float() { return std::generate_canonical<float, std::numeric_limits<float>::digits>(xorshift64star); }

inline float rand_float(float min, float max) {
    std::uniform_real_distribution<float> distr{min, max};
    return distr(xorshift64star);
}

inline double rand_double() {
    return std::generate_canonical<double, std::numeric_limits<double>::digits>(xorshift64star);
}

inline double rand_double(double min, double max) {
    std::uniform_real_distribution<double> distr{min, max};
    return distr(xorshift64star);
}

inline bool rand_bool() { return static_cast<bool>(xorshift64star() % 2); }

template <class T>
const T& rand_choise(std::initializer_list<T> objects) {
    const int random_index = rand_int(0, static_cast<int>(objects.size()) - 1);
    return objects.begin()[random_index];
}

template <class T>
T rand_linear_combination(const T& A, const T& B) { // random linear combination of 2 colors/vectors/etc
    const auto coef = rand_double();
    return A * coef + B * (1. - coef);
}

} // namespace utl::random

#endif
#endif // module utl::random






// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DmitriBogdanov/prototyping_utils ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Module:        utl::shell
// Documentation: https://github.com/DmitriBogdanov/prototyping_utils/blob/master/docs/module_shell.md
// Source repo:   https://github.com/DmitriBogdanov/prototyping_utils
//
// This project is licensed under the MIT License
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#if !defined(UTL_PICK_MODULES) || defined(UTLMODULE_SHELL)
#ifndef UTLHEADERGUARD_SHELL
#define UTLHEADERGUARD_SHELL

// _______________________ INCLUDES _______________________

#include <cstddef>       // size_t
#include <cstdlib>       // atexit(), system(), rand()
#include <filesystem>    // filesystem::remove(), filesystem::path, filesystem::exists()
#include <fstream>       // ofstream, ifstream
#include <sstream>       // ostringstream
#include <string>        // string
#include <string_view>   // string_view
#include <unordered_set> // unordered_set<>
#include <vector>        // vector<>

// ____________________ DEVELOPER DOCS ____________________

// Command line utils that allow simple creation of temporary files and command line
// calls with stdout and stderr piping (a task surprisingly untrivial in standard C++).
//
// # ::random_ascii_string() #
// Creates random ASCII string of given length.
// Uses chars in ['a', 'z'] range.
//
// # ::generate_temp_file() #
// Generates temporary .txt file with a random unique name, and returns it's filepath.
// Files generated during current runtime can be deleted with '::clear_temp_files()'.
// If '::clear_temp_files()' wasn't called manually, it gets called automatically upon exiting 'main()'.
// Uses relative path internally.
//
// # ::clear_temp_files() #
// Clears temporary files generated during current runtime.
//
// # ::erase_temp_file() #
// Clears a single temporary file with given filepath.
//
// # ::run_command() #
// Runs a command using the default system shell.
// Returns piped status (error code), stdout and stderr.
//
// # ::exe_path() #
// Parses executable path from argcv as std::string_view.
//
// # ::command_line_args() #
// Parses command line arguments from argcv as std::string_view.

// ____________________ IMPLEMENTATION ____________________

namespace utl::shell {

// =================================
// --- Temporary File Generation ---
// =================================

[[nodiscard]] inline std::string random_ascii_string(std::size_t length) {
    constexpr char min_char = 'a';
    constexpr char max_char = 'z';

    std::string result(length, '0');
    for (std::size_t i = 0; i < length; ++i)
        result[i] = static_cast<char>(min_char + rand() % (max_char - min_char + 1));
    return result;
}

inline std::unordered_set<std::string> _temp_files; // currently existing temp files
inline bool                            _temp_files_cleanup_registered = false;

inline void clear_temp_files() {
    for (const auto& file : _temp_files) std::filesystem::remove(file);
    _temp_files.clear();
}

inline void erase_temp_file(const std::string& file) {
    // we take 'file' as 'std::string&' instead of 'std::string_view' because it is
    // used to call '.erase()' on the map of 'std::string', which does not take string_view
    std::filesystem::remove(file);
    _temp_files.erase(file);
}

inline std::string generate_temp_file() {
    // No '[[nodiscard]]' since the function could still be used to generate files without
    // actually accessing them (through the returned path) in the same program.
    
    constexpr std::size_t MAX_ATTEMPTS = 500; // shouldn't realistically be encountered but still
    constexpr std::size_t NAME_LENGTH  = 30;

    // Register std::atexit() if not already registered
    if (!_temp_files_cleanup_registered) {
        const bool success             = (std::atexit(clear_temp_files) == 0);
        _temp_files_cleanup_registered = success;
    }

    // Try creating files until unique name is found
    for (std::size_t i = 0; i < MAX_ATTEMPTS; ++i) {
        const std::filesystem::path temp_path(random_ascii_string(NAME_LENGTH) + ".txt");

        if (std::filesystem::exists(temp_path)) continue;

        const std::ofstream temp_file(temp_path);

        if (temp_file.good()) {
            _temp_files.insert(temp_path.string());
            return temp_path.string();
        } else {
            return std::string();
        }
    }

    return std::string();
}

// ===================
// --- Shell Utils ---
// ===================

struct CommandResult {
    int         status; // aka error code
    std::string stdout_output;
    std::string stderr_output;
};

inline CommandResult run_command(const std::string& command) {
    // Note 1:
    // we take 'std::string&' instead of 'std::string_view' because there
    // has to be a guarantee that contained string is null-terminated

    // Note 2:
    // Creating temporary files doesn't seem to be ideal, but I'd yet to find
    // a way to pipe BOTH stdout and stderr directly into the program without
    // relying on platform-specific API like Unix forks and Windows processes

    // Note 3:
    // Usage of std::system() is often discouraged due to security reasons,
    // but it doesn't seem there is a portable way to do better (aka going
    // back to previous note about platform-specific APIs)

    const auto stdout_file = utl::shell::generate_temp_file();
    const auto stderr_file = utl::shell::generate_temp_file();

    // Redirect stdout and stderr of the command to temporary files
    std::ostringstream ss;
    ss << command.c_str() << " >" << stdout_file << " 2>" << stderr_file;
    const std::string modified_command = ss.str();

    // Call command
    const auto status = std::system(modified_command.c_str());

    // Read stdout and stderr from temp files and remove them
    std::ostringstream stdout_stream;
    std::ostringstream stderr_stream;
    stdout_stream << std::ifstream(stdout_file).rdbuf();
    stderr_stream << std::ifstream(stderr_file).rdbuf();
    utl::shell::erase_temp_file(stdout_file);
    utl::shell::erase_temp_file(stderr_file);

    // Return
    CommandResult result = {status, stdout_stream.str(), stderr_stream.str()};

    return result;
}

// =========================
// --- Argc/Argv parsing ---
// =========================

// This is just "C to C++ string conversion" for argc/argv
//
// Perhaps it could be expanded to proper parsing of standard "CLI options" format
// (like ordered/unordered flags prefixed with '--', shortcuts prefixed with '-' and etc.)

[[nodiscard]] inline std::string_view get_exe_path(char** argv) {
    // argc == 1 is a reasonable assumption since the only way to achieve such launch
    // is to run executable through a null-execv, most command-line programs assume
    // such scenario to be either impossible or an error on user side
    return std::string_view(argv[0]);
}

[[nodiscard]] inline std::vector<std::string_view> get_command_line_args(int argc, char** argv) {
    std::vector<std::string_view> arguments(argc - 1);
    for (std::size_t i = 0; i < arguments.size(); ++i) arguments.emplace_back(argv[i]);
    return arguments;
}

} // namespace utl::shell

#endif
#endif // module utl::shell






// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DmitriBogdanov/prototyping_utils ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Module:        utl::sleep
// Documentation: https://github.com/DmitriBogdanov/prototyping_utils/blob/master/docs/module_sleep.md
// Source repo:   https://github.com/DmitriBogdanov/prototyping_utils
//
// This project is licensed under the MIT License
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#if !defined(UTL_PICK_MODULES) || defined(UTLMODULE_SLEEP)
#ifndef UTLHEADERGUARD_SLEEP
#define UTLHEADERGUARD_SLEEP

// _______________________ INCLUDES _______________________

#include <chrono>  // chrono::steady_clock, chrono::nanoseconds, chrono::duration_cast<>
#include <cmath>   // sqrt()
#include <cstdint> // int64_t
#include <thread>  // this_thread::sleep_for()

// ____________________ DEVELOPER DOCS ____________________

// Various implementation of sleep(), used for precise delays.
//
// # ::spinlock() #
// Best precision, uses CPU.
//
// # ::hybrid() #
// Recommended option, similar precision to spinlock with minimal CPU usage.
// Loops short system sleep while statistically estimating its error on the fly and once within error
// margin of the end time, finished with spinlock sleep (essentialy negating usual system sleep error).
//
// # ::system() #
// Worst precision, frees CPU.

// ____________________ IMPLEMENTATION ____________________

namespace utl::sleep {
    
// =============================
// --- Sleep Implementations ---
// =============================

using _clock     = std::chrono::steady_clock;
using _chrono_ns = std::chrono::nanoseconds;

inline void spinlock(double ms) {
    const long long ns              = static_cast<std::int64_t>(ms * 1e6);
    const auto      start_timepoint = _clock::now();

    volatile int i = 0; // volatile 'i' prevents standard-compliant compilers from optimizing away the loop
    while (std::chrono::duration_cast<_chrono_ns>(_clock::now() - start_timepoint).count() < ns) { ++i; }
}

inline void hybrid(double ms) {
    static double       estimate = 5e-3; // initial sleep_for() error estimate
    static double       mean     = estimate;
    static double       m2       = 0;
    static std::int64_t count    = 1;

    // We treat sleep_for(1 ms) as a random variate "1 ms + random_value()"
    while (ms > estimate) {
        const auto start = _clock::now();
        std::this_thread::sleep_for(_chrono_ns(static_cast<std::int64_t>(1e6)));
        const auto end = _clock::now();

        const double observed = std::chrono::duration_cast<_chrono_ns>(end - start).count() / 1e6;
        ms -= observed;

        ++count;

        // Welford's algorithm for mean and unbiased variance estimation
        const double delta = observed - mean;
        mean += delta / static_cast<double>(count);
        m2 += delta * (observed - mean); // intermediate values 'm2' reduce numerical instability
        const double variance = std::sqrt(m2 / static_cast<double>(count - 1));

        estimate = mean + variance; // set estimate 1 standard deviation above the mean
        // can be adjusted to make estimate more or less pessimistic
    }

    utl::sleep::spinlock(ms);
}

inline void system(double ms) { std::this_thread::sleep_for(_chrono_ns(static_cast<std::int64_t>(ms * 1e6))); }

} // namespace utl::sleep

#endif
#endif // module utl::sleep






// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DmitriBogdanov/prototyping_utils ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Module:        utl::stre
// Documentation: https://github.com/DmitriBogdanov/prototyping_utils/blob/master/docs/module_stre.md
// Source repo:   https://github.com/DmitriBogdanov/prototyping_utils
//
// This project is licensed under the MIT License
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#include <exception>
#include <stdexcept>
#include <vector>
#if !defined(UTL_PICK_MODULES) || defined(UTLMODULE_STRE)
#ifndef UTLHEADERGUARD_STRE
#define UTLHEADERGUARD_STRE

// _______________________ INCLUDES _______________________

#include <algorithm>   // transform()
#include <cctype>      // tolower(), toupper()
#include <cstddef>     // size_t
#include <iomanip>     // setfill(), setw()
#include <ostream>     // ostream
#include <sstream>     // ostringstream
#include <string>      // string
#include <string_view> // string_view
#include <tuple>       // tuple<>, get<>()
#include <type_traits> // false_type, true_type, void_t<>, is_convertible<>, enable_if_t<>
#include <utility>     // declval<>(), index_sequence<>

// ____________________ DEVELOPER DOCS ____________________

// String utility extensions, mainly a template ::to_str() method which works with all STL containers,
// including maps, sets and tuples with any level of mutual nesting.
//
// Also includes some expansions of <type_traits> header, since we need them anyways to implement a generic ::to_str().
//
// # ::is_printable<Type> #
// Integral constant, returns in "::value" whether Type can be printed through std::cout.
// Criteria: Existance of operator 'ANY_TYPE operator<<(std::ostream&, Type&)'
//
// # ::is_iterable_through<Type> #
// Integral constant, returns in "::value" whether Type can be iterated through.
// Criteria: Existance of .begin() and .end() with applicable operator()++
//
// # ::is_const_iterable_through<Type> #
// Integral constant, returns in "::value" whether Type can be const-iterated through.
// Criteria: Existance of .cbegin() and .cend() with applicable operator()++
//
// # ::is_tuple_like<Type> #
// Integral constant, returns in "::value" whether Type has a tuple-like structure.
// Tuple-like structure include std::tuple, std::pair, std::array, std::ranges::subrange (since C++20)
// Criteria: Existance of applicable std::get<0>() and std::tuple_size()
//
// # ::is_string<Type> #
// Integral constant, returns in "::value" whether Type is a char string.
// Criteria: Type can be decayed to std::string or a char* pointer
//
// # ::is_to_str_convertible<Type> #
// Integral constant, returns in "::value" whether Type can be converted to string through ::to_str().
// Criteria: Existance of a valid utl::stre::to_str() overload
//
// # ::to_str() #
// Converts any standard container or a custom container with necessary member functions to std::string.
// Works with tuples and tuple-like classes.
// Works with nested containers/tuples through recursive template instantiation, which
// resolves as long as types at the end of recursion have a valid operator<<() for ostreams.
//
// Not particularly fast, but it doesn't really have to be since this kind of thing is mostly
// useful for debugging and other human-readable prints.
//
// # ::InlineStream #
// Inline 'std::ostringstream' construction with implicit conversion to 'std::string'.
// Rather unperformant, but convenient for using stream formating during string construction.
// Example: std::string str = (stre::InlineStream() << "Value " << 3.14 << " is smaller than " << 6.28);
//
// In retrospective the usefulness of this seems rather dubious. Perhaps I'll deprecate it later.
//
// # ::repeat_symbol(), ::repeat_string() #
// Repeats character/string a given number of times.
//
// # ::pad_with_zeroes() #
// Pads given integer with zeroes untill a certain lenght.
// Useful when saving data in files like 'data_0001.txt', 'data_0002.txt', '...' so they get properly sorted.

// ____________________ IMPLEMENTATION ____________________

namespace utl::stre {

// ================
// --- Trimming ---
// ================

template <class T>
[[nodiscard]] std::string trim_left(T&& str, char trimmed_char = ' ') {
    std::string res = std::forward<T>(str);  // when 'str' is an r-value, we can avoid the copy
    res.erase(0, res.find_first_not_of(trimmed_char)); // seems to be the fastest way of doing it
    return res;
}

template <class T>
[[nodiscard]] std::string trim_right(T&& str, char trimmed_char = ' ') {
    std::string res = std::forward<T>(str);
    res.erase(res.find_last_not_of(trimmed_char) + 1);
    return res;
}

template <class T>
[[nodiscard]] std::string trim(T&& str, char trimmed_char = ' ') {
    return trim_right(trim_left(std::forward<T>(str), trimmed_char), trimmed_char);
}

// ===============
// --- Padding ---
// ===============

[[nodiscard]] std::string pad_left(std::string_view str, std::size_t length, char padding_char = ' ') {
    if (length > str.size()) {
        std::string res;
        res.reserve(length);
        res.append(length - str.size(), padding_char);
        res += str;
        return res;
    } else return std::string(str);
}

[[nodiscard]] std::string pad_right(std::string_view str, std::size_t length, char padding_char = ' ') {
    if (length > str.size()) {
        std::string res;
        res.reserve(length);
        res += str;
        res.append(length - str.size(), padding_char);
        return res;
    } else return std::string(str);
}

[[nodiscard]] std::string pad(std::string_view str, std::size_t length, char padding_char = ' ') {
    if (length > str.size()) {
        std::string res;
        res.reserve(length);
        const std::size_t left_pad_size = (length - str.size()) / 2;
        res.append(left_pad_size, padding_char);
        res += str;
        const std::size_t right_pad_size = length - str.size() - left_pad_size;
        res.append(right_pad_size, padding_char);
        return res;
        // we try to pad evenly on both sides, but one of the pads (the right one to be exact)
        // may be a character longer than the other if the length difference is odd
    } else return std::string(str);
}

[[nodiscard]] std::string pad_with_leading_zeroes(unsigned int number, std::size_t length = 12) {
    const std::string number_str = std::to_string(number);

    if (length > number_str.size()) {
        std::string res;
        res.reserve(length);
        res.append(length - number_str.size(), '0');
        res += number_str;
        return res;
    } else return number_str;
    // we do this instead of using 'std::ostringstream' with 'std::setfill('0')' + 'std::setw()'
    // so we don't need streams as a dependency. Plus it is faster that way.
}

// ========================
// --- Case conversions ---
// ========================

template <class T>
[[nodiscard]] std::string to_lower(T&& str) {
    std::string res = std::forward<T>(str); // when 'str' is an r-value, we can avoid the copy
    std::transform(res.begin(), res.end(), res.begin(), [](unsigned char c) { return std::tolower(c); });
    return res;
    // note that 'std::tolower()', 'std::toupper()' can only apply to unsigned chars, calling it on signed char
    // is UB. Implementation above was directly taken from https://en.cppreference.com/w/cpp/string/byte/tolower
}

template <class T>
[[nodiscard]] std::string to_upper(T&& str) {
    std::string res = std::forward<T>(str);
    std::transform(res.begin(), res.end(), res.begin(), [](unsigned char c) { return std::toupper(c); });
    return res;
}

// ========================
// --- Substring checks ---
// ========================

// Note:
// C++20 adds 'std::basic_string<T>::starts_with()', 'std::basic_string<T>::ends_with()',
// 'std::basic_string<T>::contains()', making these functions pointless in a new standard.

[[nodiscard]] bool starts_with(std::string_view str, std::string_view substr) {
    return str.size() >= substr.size() && str.compare(0, substr.size(), substr) == 0;
}

[[nodiscard]] bool ends_with(std::string_view str, std::string_view substr) {
    return str.size() >= substr.size() && str.compare(str.size() - substr.size(), substr.size(), substr) == 0;
}

[[nodiscard]] bool contains(std::string_view str, std::string_view substr) {
    return str.find(substr) != std::string_view::npos;
}

// ==========================
// --- Token manipulation ---
// ==========================

template <class T>
[[nodiscard]] std::string replace_all_occurences(T&& str, std::string_view from, std::string_view to) {
    std::string res = std::forward<T>(str);

    std::size_t i = 0;
    while ((i = res.find(from, i)) != std::string::npos) { // locate substring to replace
        res.replace(i, from.size(), to);                   // replace
        i += to.size();                                    // step over the replaced region
    }
    // Note: Not stepping over the replaced regions causes self-similar replacements
    // like "abc" -> "abcabc" to fall into an infinite loop, we don't want that.

    return res;
}

// Note:
// Most "split by delimer" implementations found online seem to be horrifically inefficient
// with unnecessary copying/erasure/intermediate tokens, stringstreams and etc.
//
// We can just scan through the string view once, while keeping track of the last segment between
// two delimiters, no unnecessary work, the only place where we do a copy is during emplacement into
// the vector where it's unavoidable
[[nodiscard]] std::vector<std::string> split_by_delimiter(std::string_view str, std::string_view delimiter, bool keep_empty_tokens = false) {
    if (delimiter.empty()) return {std::string(str)};
    // handle empty delimiter explicitly so we can't fall into an infinite loop

    std::vector<std::string> tokens;
    std::size_t              cursor        = 0;
    std::size_t              segment_start = cursor;

    while ((cursor = str.find(delimiter, cursor)) != std::string_view::npos) {
        if (keep_empty_tokens || segment_start != cursor) tokens.emplace_back(str.substr(segment_start, cursor - segment_start));
        // don't emplace empty tokens in case of leading/trailing/repeated delimiter
        cursor += delimiter.size();
        segment_start = cursor;
    }

    if (keep_empty_tokens || segment_start != str.size()) tokens.emplace_back(str.substr(segment_start));
    // 'cursor' is now at 'npos', so we compare to the size instead
    
    return tokens;
}

// ===================
// --- Other utils ---
// ===================

[[nodiscard]] inline std::string repeat_char(char ch, size_t repeats) { return std::string(repeats, ch); }

[[nodiscard]] inline std::string repeat_string(std::string_view str, size_t repeats) {
    std::string res;
    res.reserve(str.size() * repeats);
    while (repeats--) res += str;
    return res;
}

// Mostly useful to print strings with special chars in console and look at their contents.
[[nodiscard]] std::string escape_control_chars(std::string_view str) {
    std::string res;
    res.reserve(str.size()); // no necesseraly correct, but it's a godd first guess

    for (const char c : str) {
        // Control characters with dedicated escape sequences get escaped with those sequences
        if (c == '\a') res += "\\a";
        else if (c == '\b') res += "\\b";
        else if (c == '\f') res += "\\f";
        else if (c == '\n') res += "\\n";
        else if (c == '\r') res += "\\r";
        else if (c == '\t') res += "\\t";
        else if (c == '\v') res += "\\v";
        // Other non-printable chars get replaced with their codes
        else if (!std::isprint(static_cast<unsigned char>(c))) {
            res += '\\';
            res += std::to_string(static_cast<int>(c));
        }
        // Printable chars are appended as is.
        else
            res += c;
    }
    // Note: This could be implemented faster using the 'utl::json' method of handling escapes with buffering and
    // a lookup table, however I don't see much practical reason to complicate this implementation like that.

    return res;
}

[[nodiscard]] std::size_t index_of_difference(std::string_view str_1, std::string_view str_2) {
    using namespace std::string_literals;
    if (str_1.size() != str_2.size())
        throw std::logic_error("String {"s + std::string(str_1) + "} of size "s + std::to_string(str_1.size()) +
                               " and {"s + std::string(str_2) + "} of size "s + std::to_string(str_2.size()) +
                               " do not have a meaningful index of difference due to incompatible sizes."s);
    for (std::size_t i = 0; i < str_1.size(); ++i)
        if (str_1[i] != str_2[i]) return i;
    return str_1.size();
}

} // namespace utl::stre

#endif
#endif // module utl::stre






// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DmitriBogdanov/prototyping_utils ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Module:        utl::table
// Documentation: https://github.com/DmitriBogdanov/prototyping_utils/blob/master/docs/module_table.md
// Source repo:   https://github.com/DmitriBogdanov/prototyping_utils
//
// This project is licensed under the MIT License
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#if !defined(UTL_PICK_MODULES) || defined(UTLMODULE_TABLE)
#ifndef UTLHEADERGUARD_TABLE
#define UTLHEADERGUARD_TABLE

// _______________________ INCLUDES _______________________

#include <cstddef>          // size_t
#include <initializer_list> // initializer_list<>
#include <iomanip>          // resetiosflags(), setw()
#include <ios>              // streamsize, ios_base::fmtflags, ios
#include <iostream>         // cout
#include <ostream>          // ostream
#include <sstream>          // ostringstream
#include <string>           // string
#include <type_traits>      // is_arithmetic_v<>, is_same_v<>
#include <vector>           // vector<>

// ____________________ DEVELOPER DOCS ____________________

// Functions used to build and render simple ASCII table in console.
//
// Tries to be simple and minimize boilerplate, exposes a LaTeX-like API.
// In fact this is for a reason - these tables can be formatted for a quick LaTeX export
// by enabling a 'set_latex_mode(true)'.
//
// As of now the implementation is short but quite frankly ugly, it feels like with some thought
// it could be generalized much better to support multiple styles and perform faster, however there
// is ~0 need for this to be fast since it's mean for human-readable tables and not massive data export.
// Adding more styles while nice also doesn't seem like an important thing as of now so the old implementation
// is left to be as is.

// ____________________ IMPLEMENTATION ____________________

namespace utl::table {

// =====================
// --- Column Format ---
// =====================

using uint       = std::streamsize;
using _ios_flags = std::ios_base::fmtflags;

struct ColumnFormat {
    _ios_flags flags;
    uint       precision;
};

struct _Column {
    uint         width;
    ColumnFormat col_format;
};

// --- Predefined Formats ---
// --------------------------

constexpr ColumnFormat NONE = {std::ios::showpoint, 6};

constexpr ColumnFormat FIXED(uint decimals = 3) { return {std::ios::fixed, decimals}; }
constexpr ColumnFormat DEFAULT(uint decimals = 6) { return {std::ios::showpoint, decimals}; }
constexpr ColumnFormat SCIENTIFIC(uint decimals = 3) { return {std::ios::scientific, decimals}; }

constexpr ColumnFormat BOOL = {std::ios::boolalpha, 3};

// --- Internal Table State ---
// ----------------------------

inline std::vector<_Column> _columns;
inline std::size_t          _current_column = 0;
inline std::ostream*        _output_stream  = &std::cout;
inline bool                 _latex_mode     = false;

// ===================
// --- Table Setup ---
// ===================

inline void create(std::initializer_list<uint>&& widths) {
    _columns.resize(widths.size());
    for (std::size_t i = 0; i < _columns.size(); ++i) {
        _columns[i].width      = widths.begin()[i];
        _columns[i].col_format = DEFAULT();
    }
}

inline void set_formats(std::initializer_list<ColumnFormat>&& formats) {
    for (std::size_t i = 0; i < _columns.size(); ++i) _columns[i].col_format = formats.begin()[i];
}

inline void set_ostream(std::ostream& new_ostream) { _output_stream = &new_ostream; }

inline void set_latex_mode(bool toggle) { _latex_mode = toggle; }

// =======================
// --- Table Rendering ---
// =======================

// We want to only apply additional typesetting to "actual mathematical numbers", not bools & chars
template <class T>
constexpr bool _is_arithmetic_number_v =
    std::is_arithmetic_v<T> && !std::is_same_v<T, bool> && !std::is_same_v<T, char>;

[[nodiscard]] std::string _trim_left(const std::string& str, char trimmed_char) {
    std::string res = str;
    res.erase(0, res.find_first_not_of(trimmed_char));
    return res;
}

// Function that adds some LaTeX decorators to appropriate types
template <class T>
void _append_decorated_value(std::ostream& os, const T& value) {
    using V = std::decay_t<T>;

    if (!_latex_mode) {
        os << value;
        return;
    }

    if constexpr (_is_arithmetic_number_v<V>) {
        // In order to respect format flags of the table, we have to copy fmt into a stringstream
        // and use IT to stringify a number, simple 'std::to_string()' won't do it here
        std::ostringstream ss;
        ss.copyfmt(os);
        ss.width(0); // cancel out 'std::setw()' that was copied with all the other flags
        ss << value;
        std::string number_string = ss.str();

        // Convert scientific form number to a LaTeX-friendly form,
        // for example, "1.3e-15" becomes "1.3 \cdot 10^{-15}"
        const std::size_t e_index = number_string.find('e');
        if (e_index != std::string::npos) {
            const std::string mantissa = number_string.substr(0, e_index - 1);
            const char        sign     = number_string.at(e_index + 1);
            const std::string exponent = number_string.substr(e_index + 2);

            number_string.clear();
            number_string += mantissa;
            number_string += " \\cdot 10^{";
            if (sign == '-') number_string += sign;
            number_string += _trim_left(exponent, '0');
            number_string += '}';
        }

        // Typeset numbers as formulas
        os << "$" << number_string << "$";
    } else os << value;
}

inline void cell(){};

template <class T, class... Types>
void cell(T value, const Types... other_values) {
    const auto left_delimer  = _latex_mode ? "" : "|";
    const auto delimer       = _latex_mode ? " & " : "|";
    const auto right_delimer = _latex_mode ? " \\\\\n" : "|\n";

    const std::string left_cline      = (_current_column == 0) ? left_delimer : "";
    const std::string right_cline     = (_current_column == _columns.size() - 1) ? right_delimer : delimer;
    const _ios_flags  format          = _columns[_current_column].col_format.flags;
    const uint        float_precision = _columns[_current_column].col_format.precision;

    // Save old stream state
    std::ios old_state(nullptr);
    old_state.copyfmt(*_output_stream);

    // Set table formatting
    (*_output_stream) << std::resetiosflags((*_output_stream).flags());
    (*_output_stream).flags(format);
    (*_output_stream).precision(float_precision);

    // Print
    (*_output_stream) << left_cline << std::setw(_columns[_current_column].width);
    _append_decorated_value(*_output_stream, value);
    (*_output_stream) << right_cline;

    // Return old stream state
    (*_output_stream).copyfmt(old_state);

    // Advance column counter
    _current_column = (_current_column == _columns.size() - 1) ? 0 : _current_column + 1;

    cell(other_values...);
}

inline void hline() {
    if (_latex_mode) {
        (*_output_stream) << "\\hline\n";
    } else {
        (*_output_stream) << "|";
        for (const auto& col : _columns)
            (*_output_stream) << std::string(static_cast<std::size_t>(col.width), '-') << "|";
        (*_output_stream) << "\n";
    }
}

} // namespace utl::table

#endif
#endif // module utl::table






// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DmitriBogdanov/prototyping_utils ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Module:        utl::timer
// Documentation: https://github.com/DmitriBogdanov/prototyping_utils/blob/master/docs/module_timer.md
// Source repo:   https://github.com/DmitriBogdanov/prototyping_utils
//
// This project is licensed under the MIT License
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#if !defined(UTL_PICK_MODULES) || defined(UTLMODULE_TIMER)
#ifndef UTLHEADERGUARD_TIMER
#define UTLHEADERGUARD_TIMER

// _______________________ INCLUDES _______________________

#include <array>   // array<>
#include <chrono>  // chrono::steady_clock, chrono::nanoseconds, chrono::duration_cast<>
#include <ctime>   // time, time_t, tm, strftime()
#include <string>  // string, to_string()
#include <utility> // forward<>()

// ____________________ DEVELOPER DOCS ____________________

// Global-state timer with built-in formatting. Functions for local date and time.
//
// Uses SFINAE to resolve platform-specific calls to local time (localtime_s() on Windows,
// localtime_r() on Linux), the same can be done with macros. The fact that there seems to
// be no portable way of getting local time before C++20 (which adds "Calendar" part of <chrono>)
// is rather bizzare, but not unmanageable.
//
// # ::start() #
// Starts the timer.
// Note: If start() wasn't called system will use uninitialized value as a start.
//
// # ::elapsed_ms(), ::elapsed_sec(), ::elapsed_min(), ::elapsed_hours() #
// Elapsed time as double.
//
// # ::elapsed_string_ms(), ::elapsed_string_sec(), ::elapsed_string_min(), ::elapsed_string_hours() #
// Elapsed time as std::string.
//
// # ::elapsed_string:fullform() #
// Elapsed time in format "%H hours %M min %S sec %MS ms".
//
// # ::datetime_string() #
// Current local date and time in format "%y-%m-%d %H:%M:%S".
//
// # ::datetime_string_id() #
// Current local date and time in format "%y-%m-%d-%H-%M-%S".
// Less readable that usual format, but can be used in filenames which prohibit ":" usage.

// ____________________ IMPLEMENTATION ____________________

namespace utl::timer {

// =================
// --- Internals ---
// =================

using _clock     = std::chrono::steady_clock;
using _chrono_ns = std::chrono::nanoseconds;

constexpr double _ns_in_ms = 1e6;

constexpr long long _ms_in_sec  = 1000;
constexpr long long _ms_in_min  = 60 * _ms_in_sec;
constexpr long long _ms_in_hour = 60 * _ms_in_min;

inline _clock::time_point _start_timepoint;

[[nodiscard]] inline double _elapsed_time_as_ms() {
    const auto elapsed = std::chrono::duration_cast<_chrono_ns>(_clock::now() - _start_timepoint).count();
    return static_cast<double>(elapsed) / _ns_in_ms;
}

inline void start() { _start_timepoint = _clock::now(); }

// ==============================
// --- Elapsed Time Functions ---
// ==============================

// --- Elapsed Time as 'double' ---
// --------------------------------

[[nodiscard]] inline double elapsed_ms() { return _elapsed_time_as_ms(); }
[[nodiscard]] inline double elapsed_sec() { return _elapsed_time_as_ms() / static_cast<double>(_ms_in_sec); }
[[nodiscard]] inline double elapsed_min() { return _elapsed_time_as_ms() / static_cast<double>(_ms_in_min); }
[[nodiscard]] inline double elapsed_hours() { return _elapsed_time_as_ms() / static_cast<double>(_ms_in_hour); }

// --- Elapsed Time as 'std::string' ---
// -------------------------------------

[[nodiscard]] inline std::string elapsed_string_ms() { return std::to_string(elapsed_ms()) + " ms"; }
[[nodiscard]] inline std::string elapsed_string_sec() { return std::to_string(elapsed_sec()) + " sec"; }
[[nodiscard]] inline std::string elapsed_string_min() { return std::to_string(elapsed_min()) + " min"; }
[[nodiscard]] inline std::string elapsed_string_hours() { return std::to_string(elapsed_hours()) + " hours"; }

[[nodiscard]] inline std::string elapsed_string_fullform() {
    long long unaccounted_ms = static_cast<long long>(_elapsed_time_as_ms());

    long long ms    = 0;
    long long min   = 0;
    long long sec   = 0;
    long long hours = 0;

    if (unaccounted_ms > _ms_in_hour) {
        hours += unaccounted_ms / _ms_in_hour;
        unaccounted_ms -= hours * _ms_in_hour;
    }

    if (unaccounted_ms > _ms_in_min) {
        min += unaccounted_ms / _ms_in_min;
        unaccounted_ms -= min * _ms_in_min;
    }

    if (unaccounted_ms > _ms_in_sec) {
        sec += unaccounted_ms / _ms_in_sec;
        unaccounted_ms -= sec * _ms_in_sec;
    }

    ms = unaccounted_ms;

    return std::to_string(hours) + " hours " + std::to_string(min) + " min " + std::to_string(sec) + " sec " +
           std::to_string(ms) + " ms ";
}

// ============================
// --- Local Time Functions ---
// ============================

// - SFINAE to select localtime_s() or localtime_r() -
template <typename Arg_tm, typename Arg_time_t>
auto _available_localtime_impl(Arg_tm time_moment, Arg_time_t timer)
    -> decltype(localtime_s(std::forward<Arg_tm>(time_moment), std::forward<Arg_time_t>(timer))) {
    return localtime_s(std::forward<Arg_tm>(time_moment), std::forward<Arg_time_t>(timer));
}

template <typename Arg_tm, typename Arg_time_t>
auto _available_localtime_impl(Arg_tm time_moment, Arg_time_t timer)
    -> decltype(localtime_r(std::forward<Arg_time_t>(timer), std::forward<Arg_tm>(time_moment))) {
    return localtime_r(std::forward<Arg_time_t>(timer), std::forward<Arg_tm>(time_moment));
}

// - Implementation -
[[nodiscard]] inline std::string _datetime_string_with_format(const char* format) {
    std::time_t timer = std::time(nullptr);
    std::tm     time_moment{};

    // Call localtime_s() or localtime_r() depending on which one is present
    _available_localtime_impl(&time_moment, &timer);

    // // Macro version, can be used instead of SFINAE resolution
    // // Get localtime safely (if possible)
    // #if defined(__unix__)
    // localtime_r(&timer, &time_moment);
    // #elif defined(_MSC_VER)
    // localtime_s(&time_moment, &timer);
    // #else
    // // static std::mutex mtx; // mutex can be used to make thread-safe version but add another dependency
    // // std::lock_guard<std::mutex> lock(mtx);
    // time_moment = *std::localtime(&timer);
    // #endif

    // Convert time to C-string
    std::array<char, 100> mbstr;
    std::strftime(mbstr.data(), mbstr.size(), format, &time_moment);

    return std::string(mbstr.data());
}

[[nodiscard]] inline std::string datetime_string() { return _datetime_string_with_format("%Y-%m-%d %H:%M:%S"); }

[[nodiscard]] inline std::string datetime_string_id() { return _datetime_string_with_format("%Y-%m-%d-%H-%M-%S"); }

} // namespace utl::timer

#endif
#endif // module utl::timer






// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DmitriBogdanov/prototyping_utils ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Module:        utl::voidstream
// Documentation: https://github.com/DmitriBogdanov/prototyping_utils/blob/master/docs/module_voidstream.md
// Source repo:   https://github.com/DmitriBogdanov/prototyping_utils
//
// This project is licensed under the MIT License
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#if !defined(UTL_PICK_MODULES) || defined(UTLMODULE_VOIDSTREAM)
#ifndef UTLHEADERGUARD_VOIDSTREAM
#define UTLHEADERGUARD_VOIDSTREAM

// _______________________ INCLUDES _______________________

#include <ostream>   // ostream
#include <streambuf> // streambuf

// ____________________ DEVELOPER DOCS ____________________

// TODO:
// Expand this module into a proper "streams" module. Being as small as it currently is
// it has rather little reason for existence. Combined with what is probably a rather
// unperformant way of discaring '::vout' inputs, it's quite justified for a rewrite.
// Streams that could be implemented:
//    - Multisink Stream (forwards inputs to multiple std::ostream's)
//    - Appending Stream (not sure if it's even implementable or useful)
//
// "voidstream" that functions like std::ostream with no output.
// Can be passed to interfaces that use streams to silence their output.
//
// # ::vstreambuf #
// Stream buffer that overflows with no output, usage example:
//   > std::ofstream output_stream(&vstreambuf);
//   > output_stream << VALUE; // produces nothing
//
// # ::vout #
// Output stream that produces no output, usage example:
//   > vout << VALUE; // produces nothing

// ____________________ IMPLEMENTATION ____________________

namespace utl::voidstream {

// ===================
// --- Void Stream ---
// ===================

class VoidStreamBuf : public std::streambuf {
public:
    inline int overflow(int c) { return c; }
};

class VoidStream : public std::ostream {
public:
    inline VoidStream() : std::ostream(&buffer) {}

private:
    VoidStreamBuf buffer;
};

inline VoidStreamBuf vstreambuf;
inline VoidStream    vout;

} // namespace utl::voidstream

#endif
#endif // module utl::voidstream






