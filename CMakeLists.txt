cmake_minimum_required(VERSION 3.15)
project(TetgenFEM CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Release 默认优化更激进
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math -DNDEBUG -mcpu=native -stdlib=libc++")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")

# 查找 OpenMP
find_package(OpenMP)

# 查找 OpenGL 和 GLFW
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

# 添加 Eigen 头文件路径
include_directories(
    ${PROJECT_SOURCE_DIR}/TetgenFEM
    ${PROJECT_SOURCE_DIR}/TetgenFEM/eigen-3.4.0
    ${PROJECT_SOURCE_DIR}/TetgenFEM/GL
    ${PROJECT_SOURCE_DIR}/TetgenFEM/GLFW
    ${OPENGL_INCLUDE_DIRS}
)

# Mac 特定的配置
if(APPLE)
    # 添加 OpenGL 框架路径
    include_directories(/System/Library/Frameworks)
    
    # 禁用 OpenGL 弃用警告
    add_definitions(-DGL_SILENCE_DEPRECATION)
    
    # 为 Eigen 禁用浮点数 infinity 警告
    add_compile_options(-Wno-nan-infinity-disabled)
endif()

# 收集源文件
set(SOURCES
    TetgenFEM/main.cpp
    TetgenFEM/Object.cpp
    TetgenFEM/Group.cpp
    TetgenFEM/Vertex.cpp
    TetgenFEM/Edge.cpp
    TetgenFEM/Tetrahedron.cpp
    TetgenFEM/VisualOpenGL.cpp
    TetgenFEM/SimpleUI.cpp
    TetgenFEM/ReadSTL.cpp
    TetgenFEM/params.cpp
    TetgenFEM/Experiment3.cpp
    TetgenFEM/Experiment1.cpp
    TetgenFEM/Experiment2.cpp
    TetgenFEM/Experiment4.cpp
)

# 创建可执行文件
add_executable(TetgenFEM ${SOURCES})

# 链接库
if(APPLE)
    # 优先使用 Homebrew LLVM 的 libc++/unwind
    link_directories(/opt/homebrew/opt/llvm/lib /opt/homebrew/opt/llvm/lib/c++ /opt/homebrew/opt/llvm/lib/unwind)
    target_link_options(TetgenFEM PRIVATE -stdlib=libc++ -L/opt/homebrew/opt/llvm/lib/c++ -L/opt/homebrew/opt/llvm/lib/unwind -lunwind)

    target_link_libraries(TetgenFEM PRIVATE
        ${OPENGL_LIBRARIES}
        GLEW::GLEW
        glfw
        ${PROJECT_SOURCE_DIR}/TetgenFEM/libtet.a
        "-framework OpenGL"
        "-framework Cocoa"
        "-framework IOKit"
        "-framework CoreVideo"
    )
else()
    target_link_libraries(TetgenFEM PRIVATE
        ${OPENGL_LIBRARIES}
        GLEW::GLEW
        glfw
        ${PROJECT_SOURCE_DIR}/TetgenFEM/tetgen.lib
    )
endif()

# 如果找到 OpenMP，则链接
if(OpenMP_CXX_FOUND)
    target_link_libraries(TetgenFEM PRIVATE OpenMP::OpenMP_CXX)
elseif(APPLE)
    # clang 在 macOS 上需要手动指定 libomp
    target_include_directories(TetgenFEM PRIVATE /opt/homebrew/opt/libomp/include)
    target_link_libraries(TetgenFEM PRIVATE /opt/homebrew/opt/libomp/lib/libomp.dylib)
    target_compile_options(TetgenFEM PRIVATE -Xpreprocessor -fopenmp)
    target_link_options(TetgenFEM PRIVATE -Xpreprocessor -fopenmp -L/opt/homebrew/opt/libomp/lib -lomp)
endif()

# 设置编译选项
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(TetgenFEM PRIVATE -O3 -Wall)
endif()

# 输出消息
message(STATUS "CMake系统: ${CMAKE_SYSTEM_NAME}")
message(STATUS "C++编译器: ${CMAKE_CXX_COMPILER}")
message(STATUS "OpenMP支持: ${OpenMP_CXX_FOUND}")
