add_executable(run main.cpp)

target_compile_features(run PRIVATE cxx_std_17)
target_compile_options(run PRIVATE -O2 -Wall -Wextra -Wpedantic -Werror)