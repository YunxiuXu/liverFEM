#file(GLOB children RELATIVE ${current} ${current}/*)
#foreach(subdir ${children})
#    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${child} AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/CMakeLists.txt)
#	    add_subdirectory(${subdir})
#    endif()
#endforeach()

################################################################################
# Utilities
################################################################################
if(VEGAFEM_BUILD_UTILITIES)
    #===========================================================================
    set(VegaFEM_utilities
        computeDistanceField
        finiteDifferenceTest
        isosurfaceMesher
        objMergeFiles
        tetMesher
    )

    foreach(exe ${VegaFEM_utilities})
      file(GLOB_RECURSE sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${exe}/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/${exe}/*.cpp")
      add_executable(${exe} ${sources})
      target_link_libraries(${exe} ${PROJECT_NAME})
      # DLL copy to install folder must be wrapped because generator expression
      # TARGET_RUNTIME_DLLS ""On non-DLL platforms, this expression always evaluates to an empty string""
      # which fails on non-DLL platforms because the custom command becomes invalid by lack of argument
      # See https://stackoverflow.com/a/73550650/4709428
      if(CMAKE_IMPORT_LIBRARY_SUFFIX) # Only on DLL platform https://cmake.org/cmake/help/latest/variable/CMAKE_IMPORT_LIBRARY_SUFFIX.html
        add_custom_command(TARGET ${exe} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${exe}> $<TARGET_FILE_DIR:${exe}> 
          COMMAND_EXPAND_LISTS
        )
        install(TARGETS ${exe}
           RUNTIME_DEPENDENCIES
           PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-"
           POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
           DIRECTORIES "$<TARGET_FILE_DIR:${exe}>"
        )
      endif()
      install(TARGETS ${exe} RUNTIME DESTINATION bin CONFIGURATIONS Release)
    endforeach()

    #===========================================================================

    set(VegaFEM_utilities
        generateInterpolant
        generateInterpolationMatrix
        generateMassMatrix
        generateSurfaceMesh
    )
    foreach(target_name ${VegaFEM_utilities})
        add_executable(${target_name} "volumetricMeshUtilities/${target_name}.cpp")
        target_link_libraries(${target_name} ${PROJECT_NAME})
        # DLL copy to install folder must be wrapped because generator expression
        # TARGET_RUNTIME_DLLS ""On non-DLL platforms, this expression always evaluates to an empty string""
        # which fails on non-DLL platforms because the custom command becomes invalid by lack of argument
        # See https://stackoverflow.com/a/73550650/4709428
        if(CMAKE_IMPORT_LIBRARY_SUFFIX) # Only on DLL platform https://cmake.org/cmake/help/latest/variable/CMAKE_IMPORT_LIBRARY_SUFFIX.html
          add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${target_name}> $<TARGET_FILE_DIR:${target_name}> 
            COMMAND_EXPAND_LISTS
          )
          install(TARGETS ${target_name}
             RUNTIME_DEPENDENCIES
             PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-"
             POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
             DIRECTORIES "$<TARGET_FILE_DIR:${target_name}>"
          )
        endif()
        install(TARGETS ${target_name} RUNTIME DESTINATION bin CONFIGURATIONS Release)
    endforeach()

    #===========================================================================

    if(VEGAFEM_BUILD_COPYLEFT)
      set(VegaFEM_utilities_gpl
          immersionMesher
          virtualTetsDriver
      )
      foreach(exe ${VegaFEM_utilities_gpl})
        file(GLOB_RECURSE sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${exe}/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/${exe}/*.cpp")
        add_executable(${exe} ${sources})
        target_link_libraries(${exe} PRIVATE vegafem)
        # DLL copy to install folder must be wrapped because generator expression
        # TARGET_RUNTIME_DLLS ""On non-DLL platforms, this expression always evaluates to an empty string""
        # which fails on non-DLL platforms because the custom command becomes invalid by lack of argument
        # See https://stackoverflow.com/a/73550650/4709428
        if(CMAKE_IMPORT_LIBRARY_SUFFIX) # Only on DLL platform https://cmake.org/cmake/help/latest/variable/CMAKE_IMPORT_LIBRARY_SUFFIX.html
          add_custom_command(TARGET ${exe} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${exe}> $<TARGET_FILE_DIR:${exe}> 
            COMMAND_EXPAND_LISTS
          )
          install(TARGETS ${exe}
             RUNTIME_DEPENDENCIES
             PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-"
             POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
             DIRECTORIES "$<TARGET_FILE_DIR:${exe}>"
          )
        endif()
        install(TARGETS ${exe} RUNTIME DESTINATION bin CONFIGURATIONS Release)
      endforeach()
    endif()
endif()

################################################################################
# Utilities graphics
################################################################################
if(VEGAFEM_BUILD_UTILITIES_GRAPHICS)
    option(VEGAFEM_BUILD_WX "Build wxWidgets-based utilities (largeModalDeformationFactory)" OFF)
    set(VEGAFEM_X11_LIB "")
    if(UNIX AND NOT APPLE)
      find_package(X11 REQUIRED) # freeglut on Linux
      set(VEGAFEM_X11_LIB X11::X11)
    endif()
    set(VEGAFEM_GLUI_TARGET "")
    find_path(GLUI_INCLUDE_DIR GL/glui.h PATH_SUFFIXES GL include PATHS /usr/local/include /opt/homebrew/include)
    find_library(GLUI_LIBRARY NAMES glui PATHS /usr/local/lib /opt/homebrew/lib)
    if(GLUI_INCLUDE_DIR AND GLUI_LIBRARY)
      add_library(glui::glui UNKNOWN IMPORTED)
      set_target_properties(glui::glui PROPERTIES
        IMPORTED_LOCATION ${GLUI_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${GLUI_INCLUDE_DIR}
      )
      set(VEGAFEM_GLUI_TARGET glui::glui)
    else()
      find_package(glui CONFIG)
      if(TARGET glui::glui)
        set(VEGAFEM_GLUI_TARGET glui::glui)
      elseif(TARGET glui::glui_static)
        set(VEGAFEM_GLUI_TARGET glui::glui_static)
      endif()
    endif()
    if(NOT VEGAFEM_GLUI_TARGET)
      message(FATAL_ERROR "GLUI not found. Install glui (e.g. via Homebrew or vcpkg).")
    endif()

    #===========================================================================

    set(VegaFEM_utilities
        clothBW-rt
        displayObj
        editShapeARAP
        interactiveDeformableSimulator
        # largeModalDeformationFactory
        reducedDynamicSolver-rt
    )
    if(NOT VEGAFEM_USE_MKL)
      list(REMOVE_ITEM VegaFEM_utilities editShapeARAP reducedDynamicSolver-rt)
    endif()
    foreach(target_name ${VegaFEM_utilities})
      file(GLOB_RECURSE sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS "${target_name}/*.c" "${target_name}/*.cpp")
      add_executable(${target_name} ${sources})
      target_link_libraries(${target_name} PRIVATE ${PROJECT_NAME} 
          OpenGL::GL 
          GLUT::GLUT
          ${VEGAFEM_GLUI_TARGET} #reducedDynamicSolver-rt interactiveDeformableSimulator editShapeARAP displayObj clothBW-rt
          ${VEGAFEM_X11_LIB}
      )
      # DLL copy to install folder must be wrapped with condition because generator expression
      # TARGET_RUNTIME_DLLS ""On non-DLL platforms, this expression always evaluates to an empty string""
      # which fails on non-DLL platforms because the custom command becomes invalid by lack of argument
      # See https://stackoverflow.com/a/73550650/4709428
      if(CMAKE_IMPORT_LIBRARY_SUFFIX) # Only on DLL platform https://cmake.org/cmake/help/latest/variable/CMAKE_IMPORT_LIBRARY_SUFFIX.html
        add_custom_command(TARGET ${target_name} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${target_name}> $<TARGET_FILE_DIR:${target_name}> 
          COMMAND_EXPAND_LISTS
        )
        install(TARGETS ${target_name}
           RUNTIME_DEPENDENCIES
           PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-"
           POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
           DIRECTORIES "$<TARGET_FILE_DIR:${target_name}>"
       )
      endif()
      install(TARGETS ${target_name} RUNTIME DESTINATION bin CONFIGURATIONS Release)
    endforeach()
    target_link_libraries(displayObj PRIVATE GLEW::GLEW)

    #===========================================================================

    if(VEGAFEM_BUILD_WX)
      set(exe largeModalDeformationFactory) # Requires wxwidget
      file(GLOB_RECURSE sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${exe}/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/${exe}/*.cpp")
      add_executable(${exe} WIN32 ${sources})
      target_link_libraries(${exe} PRIVATE ${PROJECT_NAME} 
          OpenGL::GL 
          GLUT::GLUT
          ${VEGAFEM_X11_LIB}
      )
      find_package(wxWidgets QUIET COMPONENTS gl core base)
      if(wxWidgets_FOUND)
        if(TARGET wx::core)
          target_link_libraries(largeModalDeformationFactory PRIVATE wx::core wx::base wx::gl)
        else()
          if(wxWidgets_USE_FILE)
            include(${wxWidgets_USE_FILE})
          endif()
          target_link_libraries(largeModalDeformationFactory PRIVATE ${wxWidgets_LIBRARIES})
        endif()
        target_link_libraries(largeModalDeformationFactory PRIVATE GLEW::GLEW)
      else()
        message(WARNING "wxWidgets not found; skipping largeModalDeformationFactory.")
        set_target_properties(${exe} PROPERTIES EXCLUDE_FROM_ALL TRUE EXCLUDE_FROM_DEFAULT_BUILD TRUE)
      endif()

      if(CMAKE_IMPORT_LIBRARY_SUFFIX) # Only on DLL platform https://cmake.org/cmake/help/latest/variable/CMAKE_IMPORT_LIBRARY_SUFFIX.html
        add_custom_command(TARGET ${exe} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${exe}> $<TARGET_FILE_DIR:${exe}> 
          COMMAND_EXPAND_LISTS
        )
        install(TARGETS ${exe}
           RUNTIME_DEPENDENCIES
           PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-"
           POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
           DIRECTORIES "$<TARGET_FILE_DIR:${exe}>"
        )
      endif()
      install(TARGETS ${exe} RUNTIME DESTINATION bin CONFIGURATIONS Release)
    else()
      message(STATUS "Skipping largeModalDeformationFactory (VEGAFEM_BUILD_WX=OFF).")
    endif()
endif()
