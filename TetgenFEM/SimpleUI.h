#pragma once

#include <string>

struct GLFWwindow;

namespace SimpleUI {

struct Rect {
	float x = 0.0f; // window coordinates (top-left origin, y down)
	float y = 0.0f;
	float w = 0.0f;
	float h = 0.0f;
};

struct PanelLayout {
	float margin = 12.0f;
	float width = 280.0f;
	float height = 240.0f;
};

// Used by input callbacks to avoid rotating the camera while clicking UI.
bool IsCursorInDefaultPanel(GLFWwindow* window, double cursorX, double cursorY);

struct FrameState {
	int windowWidth = 1;
	int windowHeight = 1;
	int framebufferWidth = 1;
	int framebufferHeight = 1;
	float scaleX = 1.0f;
	float scaleY = 1.0f;

	double mouseXWindow = 0.0;
	double mouseYWindow = 0.0;
	double mouseXFramebuffer = 0.0;
	double mouseYFramebuffer = 0.0;

	bool mouseDown = false;
	bool mousePressed = false;  // went up->down this frame
	bool mouseReleased = false; // went down->up this frame
};

class Context {
public:
	void beginFrame(GLFWwindow* window);
	void beginDraw2D() const;
	void endDraw2D() const;

	// Interaction-only helpers (no drawing).
	bool clicked(const Rect& rect, bool enabled = true) const;
	bool held(const Rect& rect, bool enabled = true) const;

	// Drawing-only helper (uses current mouse state for hover/active style).
	void renderButton(const Rect& rect, const std::string& label, bool enabled = true) const;

	// Returns true only on click (press inside rect).
	bool button(const Rect& rect, const std::string& label, bool enabled = true) const;
	// Returns true while held inside rect.
	bool holdButton(const Rect& rect, const std::string& label, bool enabled = true) const;

	// Helper drawing primitives; rect coordinates are in window units.
	void drawPanelBackground(const Rect& rect) const;
	void drawLabel(const Rect& rect, const std::string& label, float sizePx) const;

	const FrameState& state() const { return state_; }

private:
	static bool pointInRect(double x, double y, const Rect& rect);
	Rect toFramebufferRect(const Rect& rect) const;

	void drawButton(const Rect& rect, const std::string& label, bool hot, bool active, bool enabled) const;

	FrameState state_{};
	bool prevMouseDown_ = false;
};

} // namespace SimpleUI
