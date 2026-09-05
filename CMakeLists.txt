cmake_minimum_required(VERSION 3.15)
project(TetgenFEM CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if(WIN32)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
endif()

# Release flags: keep Apple/Homebrew LLVM flags off MSVC.
if(APPLE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -mcpu=native -stdlib=libc++")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
endif()

find_package(OpenMP)
find_package(OpenGL REQUIRED)

# GLFW / GLEW: system packages on macOS; FetchContent fallback on Windows.
set(TETFEM_GLFW_TARGET glfw)
set(TETFEM_GLEW_TARGET GLEW::GLEW)
set(TETFEM_FETCHED_GLEW OFF)

if(WIN32)
    find_package(glfw3 QUIET)
    find_package(GLEW QUIET)
    include(FetchContent)
    if(NOT TARGET glfw AND NOT TARGET glfw3)
        FetchContent_Declare(
            glfw
            GIT_REPOSITORY https://github.com/glfw/glfw.git
            GIT_TAG 3.3.10
            GIT_SHALLOW TRUE
        )
        set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(glfw)
    endif()
    if(TARGET glfw)
        set(TETFEM_GLFW_TARGET glfw)
    elseif(TARGET glfw3)
        set(TETFEM_GLFW_TARGET glfw3)
    endif()

    if(NOT TARGET GLEW::GLEW)
        FetchContent_Declare(
            glew
            URL https://github.com/nigels-com/glew/releases/download/glew-2.2.0/glew-2.2.0.tgz
        )
        FetchContent_GetProperties(glew)
        if(NOT glew_POPULATED)
            FetchContent_Populate(glew)
            add_library(glew_s STATIC "${glew_SOURCE_DIR}/src/glew.c")
            target_include_directories(glew_s PUBLIC "${glew_SOURCE_DIR}/include")
            target_compile_definitions(glew_s PUBLIC GLEW_STATIC)
            target_link_libraries(glew_s PUBLIC OpenGL::GL)
        endif()
        set(TETFEM_GLEW_TARGET glew_s)
        set(TETFEM_FETCHED_GLEW ON)
    else()
        set(TETFEM_GLEW_TARGET GLEW::GLEW)
    endif()
else()
    find_package(GLEW REQUIRED)
    find_package(glfw3 REQUIRED)
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}/TetgenFEM
    ${PROJECT_SOURCE_DIR}/TetgenFEM/eigen-3.4.0
    ${OPENGL_INCLUDE_DIRS}
)
if(NOT TETFEM_FETCHED_GLEW)
    include_directories(${PROJECT_SOURCE_DIR}/TetgenFEM/GL)
endif()
if(NOT WIN32)
    include_directories(${PROJECT_SOURCE_DIR}/TetgenFEM/GLFW)
endif()

if(APPLE)
    include_directories(/System/Library/Frameworks)
    add_definitions(-DGL_SILENCE_DEPRECATION)
    add_compile_options(-Wno-nan-infinity-disabled)
endif()

# Build TetGen from source so Windows does not depend on a prebuilt .lib/.a ABI.
add_library(tet STATIC
    tetgen1.6.0/tetgen.cxx
    tetgen1.6.0/predicates.cxx
)
target_compile_definitions(tet PUBLIC TETLIBRARY)
target_include_directories(tet PUBLIC
    ${PROJECT_SOURCE_DIR}/tetgen1.6.0
    ${PROJECT_SOURCE_DIR}/TetgenFEM
)
if(MSVC)
    target_compile_options(tet PRIVATE /bigobj /wd4244 /wd4267 /wd4996)
endif()

set(SOURCES
    TetgenFEM/main.cpp
    TetgenFEM/Object.cpp
    TetgenFEM/Group.cpp
    TetgenFEM/Vertex.cpp
    TetgenFEM/Edge.cpp
    TetgenFEM/Tetrahedron.cpp
    TetgenFEM/VisualOpenGL.cpp
    TetgenFEM/SimpleUI.cpp
    TetgenFEM/AgentSphereContact.cpp
    TetgenFEM/ReadSTL.cpp
    TetgenFEM/params.cpp
    TetgenFEM/Experiment3.cpp
    TetgenFEM/Experiment1.cpp
    TetgenFEM/Experiment2.cpp
    TetgenFEM/Experiment4.cpp
    TetgenFEM/HapticInterface.cpp
)

add_executable(TetgenFEM ${SOURCES})
if(MSVC)
    target_compile_options(TetgenFEM PRIVATE /bigobj /utf-8 /wd4244 /wd4267 /wd4996 /openmp:llvm)
endif()
# Prefer FetchContent / system GLFW+GLEW headers over the copies under TetgenFEM/.
target_include_directories(TetgenFEM BEFORE PRIVATE
    $<TARGET_PROPERTY:${TETFEM_GLFW_TARGET},INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:${TETFEM_GLEW_TARGET},INTERFACE_INCLUDE_DIRECTORIES>
)

set(ULTRALEAP_DIR "${PROJECT_SOURCE_DIR}/Ultraleap")
set(TETFEM_LEAPC_ENABLED OFF)

if(WIN32)
    set(_leap_inc "${ULTRALEAP_DIR}")
    set(_leap_lib "${ULTRALEAP_DIR}/LeapC.lib")
    set(_leap_dll "${ULTRALEAP_DIR}/LeapC.dll")
    if(EXISTS "${_leap_inc}/LeapC.h" AND EXISTS "${_leap_lib}")
        target_compile_definitions(TetgenFEM PRIVATE TETFEM_HAVE_LEAPC=1)
        target_include_directories(TetgenFEM PRIVATE "${_leap_inc}")
        target_link_libraries(TetgenFEM PRIVATE "${_leap_lib}")
        if(EXISTS "${_leap_dll}")
            add_custom_command(TARGET TetgenFEM POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_leap_dll}"
                        "$<TARGET_FILE_DIR:TetgenFEM>/LeapC.dll")
        endif()
        set(TETFEM_LEAPC_ENABLED ON)
        message(STATUS "LeapC: bundled ${_leap_lib}")
    else()
        set(_ultraleap_path_root "$ENV{ProgramFiles}/Ultraleap")
        find_package(LeapSDK 5 QUIET
            PATHS
                "${_ultraleap_path_root}"
                "${_ultraleap_path_root}/LeapSDK"
        )
        if(TARGET LeapSDK::LeapC)
            set_property(TARGET LeapSDK::LeapC PROPERTY MAP_IMPORTED_CONFIG_RELEASE RELWITHDEBINFO)
            set_property(TARGET LeapSDK::LeapC PROPERTY MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELWITHDEBINFO)
            set_property(TARGET LeapSDK::LeapC PROPERTY MAP_IMPORTED_CONFIG_MINSIZEREL RELWITHDEBINFO)
            set_property(TARGET LeapSDK::LeapC PROPERTY MAP_IMPORTED_CONFIG_DEBUG RELWITHDEBINFO)
            target_compile_definitions(TetgenFEM PRIVATE TETFEM_HAVE_LEAPC=1)
            target_link_libraries(TetgenFEM PRIVATE LeapSDK::LeapC)
            get_target_property(_leap_cfgs LeapSDK::LeapC IMPORTED_CONFIGURATIONS)
            list(GET _leap_cfgs 0 _leap_cfg)
            string(TOUPPER "${_leap_cfg}" _leap_cfg_u)
            get_target_property(_leap_dll LeapSDK::LeapC IMPORTED_LOCATION_${_leap_cfg_u})
            if(_leap_dll)
                add_custom_command(TARGET TetgenFEM POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            "${_leap_dll}"
                            "$<TARGET_FILE_DIR:TetgenFEM>/LeapC.dll")
            endif()
            set(TETFEM_LEAPC_ENABLED ON)
            message(STATUS "LeapC: LeapSDK::LeapC")
        endif()
    endif()
elseif(APPLE)
    set(ULTRALEAP_LEAPC_LIB "")
    if(EXISTS "${ULTRALEAP_DIR}/libLeapC.6.dylib")
        set(ULTRALEAP_LEAPC_LIB "${ULTRALEAP_DIR}/libLeapC.6.dylib")
    elseif(EXISTS "${ULTRALEAP_DIR}/libLeapC.dylib")
        set(ULTRALEAP_LEAPC_LIB "${ULTRALEAP_DIR}/libLeapC.dylib")
    endif()
    if(EXISTS "${ULTRALEAP_DIR}/LeapC.h" AND ULTRALEAP_LEAPC_LIB)
        target_compile_definitions(TetgenFEM PRIVATE TETFEM_HAVE_LEAPC=1)
        target_include_directories(TetgenFEM PRIVATE "${ULTRALEAP_DIR}")
        target_link_libraries(TetgenFEM PRIVATE "${ULTRALEAP_LEAPC_LIB}")
        target_link_options(TetgenFEM PRIVATE "-Wl,-rpath,@executable_path")
        add_custom_command(TARGET TetgenFEM POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ULTRALEAP_LEAPC_LIB}"
                    "$<TARGET_FILE_DIR:TetgenFEM>/libLeapC.6.dylib")
        set(TETFEM_LEAPC_ENABLED ON)
    endif()
endif()

if(NOT TETFEM_LEAPC_ENABLED)
    target_compile_definitions(TetgenFEM PRIVATE TETFEM_HAVE_LEAPC=0)
    message(STATUS "LeapC: disabled")
endif()

if(WIN32)
    set(_tetfem_dist "${PROJECT_SOURCE_DIR}/TetgenFEM")
    add_custom_command(TARGET TetgenFEM POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:TetgenFEM>"
                "${_tetfem_dist}/TetgenFEM.exe"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ULTRALEAP_DIR}/LeapC.dll"
                "${_tetfem_dist}/LeapC.dll"
        COMMENT "Copy portable Windows binaries into TetgenFEM/")
endif()

if(APPLE)
    link_directories(/opt/homebrew/opt/llvm/lib /opt/homebrew/opt/llvm/lib/c++ /opt/homebrew/opt/llvm/lib/unwind)
    target_link_options(TetgenFEM PRIVATE -stdlib=libc++ -L/opt/homebrew/opt/llvm/lib/c++ -L/opt/homebrew/opt/llvm/lib/unwind -lunwind)
    target_link_libraries(TetgenFEM PRIVATE
        ${OPENGL_LIBRARIES}
        ${TETFEM_GLEW_TARGET}
        ${TETFEM_GLFW_TARGET}
        tet
        "-framework OpenGL"
        "-framework Cocoa"
        "-framework IOKit"
        "-framework CoreVideo"
    )
elseif(WIN32)
    target_link_libraries(TetgenFEM PRIVATE
        OpenGL::GL
        ${TETFEM_GLEW_TARGET}
        ${TETFEM_GLFW_TARGET}
        tet
        glu32
        gdi32
        user32
        opengl32
    )
else()
    target_link_libraries(TetgenFEM PRIVATE
        ${OPENGL_LIBRARIES}
        ${TETFEM_GLEW_TARGET}
        ${TETFEM_GLFW_TARGET}
        tet
    )
endif()

if(MSVC)
    # Classic MSVC OpenMP 2.0 does not support reduction(max:).
    target_compile_options(TetgenFEM PRIVATE /openmp:llvm)
elseif(OpenMP_CXX_FOUND)
    target_link_libraries(TetgenFEM PRIVATE OpenMP::OpenMP_CXX)
elseif(APPLE)
    target_include_directories(TetgenFEM PRIVATE /opt/homebrew/opt/libomp/include)
    target_link_libraries(TetgenFEM PRIVATE /opt/homebrew/opt/libomp/lib/libomp.dylib)
    target_compile_options(TetgenFEM PRIVATE -Xpreprocessor -fopenmp)
    target_link_options(TetgenFEM PRIVATE -Xpreprocessor -fopenmp -L/opt/homebrew/opt/libomp/lib -lomp)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(TetgenFEM PRIVATE -O3 -Wall)
endif()

message(STATUS "CMake系统: ${CMAKE_SYSTEM_NAME}")
message(STATUS "C++编译器: ${CMAKE_CXX_COMPILER}")
message(STATUS "OpenMP支持: ${OpenMP_CXX_FOUND}")
message(STATUS "LeapC启用: ${TETFEM_LEAPC_ENABLED}")
message(STATUS "GLFW target: ${TETFEM_GLFW_TARGET}")
message(STATUS "GLEW target: ${TETFEM_GLEW_TARGET}")
