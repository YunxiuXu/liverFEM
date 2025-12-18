// dear imgui: Renderer Backend for OpenGL2 (legacy OpenGL, fixed pipeline)
// This is a lightweight backend intended for OpenGL 2.1 / compatibility contexts.
// It is provided to support platforms where a modern core-profile context is not available.
//
// Build with: add this .cpp/.h and link with your OpenGL loader / headers.
// Needs to be used along with a Platform Backend (e.g. GLFW).

#pragma once
#include "imgui.h" // IMGUI_IMPL_API

IMGUI_IMPL_API bool     ImGui_ImplOpenGL2_Init();
IMGUI_IMPL_API void     ImGui_ImplOpenGL2_Shutdown();
IMGUI_IMPL_API void     ImGui_ImplOpenGL2_NewFrame();
IMGUI_IMPL_API void     ImGui_ImplOpenGL2_RenderDrawData(ImDrawData* draw_data);

// (Optional) Called by Init/NewFrame/Shutdown
IMGUI_IMPL_API bool     ImGui_ImplOpenGL2_CreateFontsTexture();
IMGUI_IMPL_API void     ImGui_ImplOpenGL2_DestroyFontsTexture();
