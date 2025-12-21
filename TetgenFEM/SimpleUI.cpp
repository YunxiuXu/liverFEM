#include "SimpleUI.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>

#include "GL/glew.h"
#include "GLFW/glfw3.h"

namespace SimpleUI {
namespace {

[[maybe_unused]] constexpr PanelLayout kDefaultPanel{};

[[maybe_unused]] Rect defaultPanelRect() {
	return Rect{ kDefaultPanel.margin, kDefaultPanel.margin, kDefaultPanel.width, kDefaultPanel.height };
}

// A tiny "segment" font (enough for short uppercase labels).
enum Segment : unsigned int {
	SegTop = 1u << 0,
	SegUpperRight = 1u << 1,
	SegLowerRight = 1u << 2,
	SegBottom = 1u << 3,
	SegLowerLeft = 1u << 4,
	SegUpperLeft = 1u << 5,
	SegMiddle = 1u << 6,
	SegDiagTLBR = 1u << 7,
	SegDiagBLTR = 1u << 8,
	SegCenter = 1u << 9,
};

unsigned int glyphSegments(char c) {
	switch (c) {
	case '0': return SegTop | SegUpperRight | SegLowerRight | SegBottom | SegLowerLeft | SegUpperLeft;
	case '1': return SegUpperRight | SegLowerRight;
	case '2': return SegTop | SegUpperRight | SegMiddle | SegLowerLeft | SegBottom;
	case '3': return SegTop | SegUpperRight | SegMiddle | SegLowerRight | SegBottom;
	case '4': return SegUpperLeft | SegMiddle | SegUpperRight | SegLowerRight;
	case '5': return SegTop | SegUpperLeft | SegMiddle | SegLowerRight | SegBottom;
	case '6': return SegTop | SegUpperLeft | SegMiddle | SegLowerRight | SegLowerLeft | SegBottom;
	case '7': return SegTop | SegUpperRight | SegLowerRight;
	case '8': return SegTop | SegUpperRight | SegLowerRight | SegBottom | SegLowerLeft | SegUpperLeft | SegMiddle;
	case '9': return SegTop | SegUpperRight | SegLowerRight | SegBottom | SegUpperLeft | SegMiddle;

	case 'A': return SegTop | SegUpperLeft | SegUpperRight | SegLowerLeft | SegLowerRight | SegMiddle;
	case 'B': return SegTop | SegUpperRight | SegLowerRight | SegBottom | SegLowerLeft | SegUpperLeft | SegMiddle;
	case 'C': return SegTop | SegUpperLeft | SegLowerLeft | SegBottom;
	case 'D': return SegTop | SegUpperRight | SegLowerRight | SegBottom | SegUpperLeft | SegLowerLeft;
	case 'E': return SegTop | SegUpperLeft | SegLowerLeft | SegBottom | SegMiddle;
	case 'F': return SegTop | SegUpperLeft | SegLowerLeft | SegMiddle;
	case 'G': return SegTop | SegUpperLeft | SegLowerLeft | SegBottom | SegLowerRight | SegMiddle;
	case 'H': return SegUpperLeft | SegLowerLeft | SegUpperRight | SegLowerRight | SegMiddle;
	case 'I': return SegTop | SegBottom | SegCenter;
	case 'J': return SegUpperRight | SegLowerRight | SegBottom | SegLowerLeft;
	case 'K': return SegUpperLeft | SegLowerLeft | SegMiddle | SegDiagBLTR | SegDiagTLBR;
	case 'L': return SegUpperLeft | SegLowerLeft | SegBottom;
	case 'M': return SegUpperLeft | SegLowerLeft | SegUpperRight | SegLowerRight | SegTop | SegCenter;
	case 'N': return SegUpperLeft | SegLowerLeft | SegUpperRight | SegLowerRight | SegDiagTLBR;
	case 'O': return SegTop | SegUpperRight | SegLowerRight | SegBottom | SegLowerLeft | SegUpperLeft;
	case 'P': return SegTop | SegUpperLeft | SegUpperRight | SegMiddle | SegLowerLeft;
	case 'Q': return SegTop | SegUpperLeft | SegUpperRight | SegLowerLeft | SegLowerRight | SegBottom | SegDiagTLBR;
	case 'R': return SegTop | SegUpperLeft | SegUpperRight | SegMiddle | SegLowerLeft | SegDiagTLBR;
	case 'S': return SegTop | SegUpperLeft | SegMiddle | SegLowerRight | SegBottom;
	case 'T': return SegTop | SegCenter;
	case 'U': return SegUpperLeft | SegLowerLeft | SegBottom | SegUpperRight | SegLowerRight;
	case 'V': return SegUpperLeft | SegLowerLeft | SegDiagBLTR;
	case 'W': return SegUpperLeft | SegLowerLeft | SegUpperRight | SegLowerRight | SegBottom | SegCenter;
	case 'X': return SegDiagTLBR | SegDiagBLTR;
	case 'Y': return SegDiagTLBR | SegDiagBLTR | SegCenter;
	case 'Z': return SegTop | SegDiagBLTR | SegBottom;
	case '-': return SegMiddle;
	case '>': return SegDiagTLBR; // crude arrow
	case '<': return SegDiagBLTR; // crude arrow
	case ' ': return 0;
	default: return 0;
	}
}

void drawSegments(unsigned int mask, float x, float y, float size, float lineWidth) {
	const float w = size;
	const float h = size * 1.6f;
	const float midY = y + h * 0.5f;
	const float leftX = x;
	const float rightX = x + w;
	const float topY = y;
	const float bottomY = y + h;

	glLineWidth(lineWidth);
	glBegin(GL_LINES);
	if (mask & SegTop) { glVertex2f(leftX, topY); glVertex2f(rightX, topY); }
	if (mask & SegUpperRight) { glVertex2f(rightX, topY); glVertex2f(rightX, midY); }
	if (mask & SegLowerRight) { glVertex2f(rightX, midY); glVertex2f(rightX, bottomY); }
	if (mask & SegBottom) { glVertex2f(leftX, bottomY); glVertex2f(rightX, bottomY); }
	if (mask & SegLowerLeft) { glVertex2f(leftX, midY); glVertex2f(leftX, bottomY); }
	if (mask & SegUpperLeft) { glVertex2f(leftX, topY); glVertex2f(leftX, midY); }
	if (mask & SegMiddle) { glVertex2f(leftX, midY); glVertex2f(rightX, midY); }
	if (mask & SegDiagTLBR) { glVertex2f(leftX, topY); glVertex2f(rightX, bottomY); }
	if (mask & SegDiagBLTR) { glVertex2f(leftX, bottomY); glVertex2f(rightX, topY); }
	if (mask & SegCenter) { glVertex2f(x + w * 0.5f, topY); glVertex2f(x + w * 0.5f, bottomY); }
	glEnd();
}

float approximateTextWidth(const std::string& text, float sizePx) {
	// Each glyph occupies (sizePx + spacing).
	const float spacing = sizePx * 0.5f; // Increased spacing for better readability
	return static_cast<float>(text.size()) * (sizePx + spacing);
}

} // namespace

bool IsCursorInDefaultPanel(GLFWwindow* window, double cursorX, double cursorY) {
	// UI is currently display-only (no interactive controls), so it should never
	// block mouse interactions in the 3D view (camera rotate/drag/etc.).
	(void)window;
	(void)cursorX;
	(void)cursorY;
	return false;
}

bool Context::pointInRect(double x, double y, const Rect& rect) {
	return x >= rect.x && x <= (rect.x + rect.w) && y >= rect.y && y <= (rect.y + rect.h);
}

Rect Context::toFramebufferRect(const Rect& rect) const {
	return Rect{
		static_cast<float>(rect.x * state_.scaleX),
		static_cast<float>(rect.y * state_.scaleY),
		static_cast<float>(rect.w * state_.scaleX),
		static_cast<float>(rect.h * state_.scaleY),
	};
}

void Context::beginFrame(GLFWwindow* window) {
	int winW = 1, winH = 1;
	glfwGetWindowSize(window, &winW, &winH);
	int fbW = 1, fbH = 1;
	glfwGetFramebufferSize(window, &fbW, &fbH);
	winW = (winW == 0) ? 1 : winW;
	winH = (winH == 0) ? 1 : winH;
	fbW = (fbW == 0) ? 1 : fbW;
	fbH = (fbH == 0) ? 1 : fbH;

	state_.windowWidth = winW;
	state_.windowHeight = winH;
	state_.framebufferWidth = fbW;
	state_.framebufferHeight = fbH;
	state_.scaleX = static_cast<float>(fbW) / static_cast<float>(winW);
	state_.scaleY = static_cast<float>(fbH) / static_cast<float>(winH);

	glfwGetCursorPos(window, &state_.mouseXWindow, &state_.mouseYWindow);
	state_.mouseXFramebuffer = state_.mouseXWindow * state_.scaleX;
	state_.mouseYFramebuffer = state_.mouseYWindow * state_.scaleY;

	const bool down = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
	state_.mouseDown = down;
	state_.mousePressed = down && !prevMouseDown_;
	state_.mouseReleased = !down && prevMouseDown_;
	prevMouseDown_ = down;
}

void Context::beginDraw2D() const {
	glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_LINE_BIT);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0.0, state_.framebufferWidth, state_.framebufferHeight, 0.0, -1.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
}

void Context::endDraw2D() const {
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glPopAttrib();
}

void Context::drawPanelBackground(const Rect& rect) const {
	const Rect r = toFramebufferRect(rect);
	glColor4f(0.05f, 0.05f, 0.06f, 0.75f);
	glBegin(GL_QUADS);
	glVertex2f(r.x, r.y);
	glVertex2f(r.x + r.w, r.y);
	glVertex2f(r.x + r.w, r.y + r.h);
	glVertex2f(r.x, r.y + r.h);
	glEnd();

	glColor4f(1.0f, 1.0f, 1.0f, 0.18f);
	glBegin(GL_LINE_LOOP);
	glVertex2f(r.x, r.y);
	glVertex2f(r.x + r.w, r.y);
	glVertex2f(r.x + r.w, r.y + r.h);
	glVertex2f(r.x, r.y + r.h);
	glEnd();
}

void Context::drawLabel(const Rect& rect, const std::string& label, float sizePx) const {
	const Rect r = toFramebufferRect(rect);
	const float size = sizePx * state_.scaleX;
	const float labelW = approximateTextWidth(label, size);
	const float x = r.x + (r.w - labelW) * 0.5f;
	const float y = r.y + (r.h - size * 1.6f) * 0.5f;

	glColor4f(1.0f, 1.0f, 1.0f, 0.92f);
	float penX = x;
	const float spacing = size * 0.5f; // Match the updated approximateTextWidth
	for (char c : label) {
		unsigned int segs = glyphSegments(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
		if (segs != 0) {
			drawSegments(segs, penX, y, size, std::max(1.0f, size * 0.08f)); // Slightly thinner lines
		}
		penX += size + spacing;
	}
}

void Context::drawButton(const Rect& rect, const std::string& label, bool hot, bool active, bool enabled) const {
	const Rect r = toFramebufferRect(rect);

	float bg = enabled ? 0.16f : 0.10f;
	float alpha = enabled ? 0.86f : 0.55f;
	if (hot && enabled) bg = 0.22f;
	if (active && enabled) bg = 0.30f;

	glColor4f(bg, bg, bg + 0.02f, alpha);
	glBegin(GL_QUADS);
	glVertex2f(r.x, r.y);
	glVertex2f(r.x + r.w, r.y);
	glVertex2f(r.x + r.w, r.y + r.h);
	glVertex2f(r.x, r.y + r.h);
	glEnd();

	glColor4f(1.0f, 1.0f, 1.0f, enabled ? 0.20f : 0.12f);
	glBegin(GL_LINE_LOOP);
	glVertex2f(r.x, r.y);
	glVertex2f(r.x + r.w, r.y);
	glVertex2f(r.x + r.w, r.y + r.h);
	glVertex2f(r.x, r.y + r.h);
	glEnd();

	drawLabel(rect, label, std::max(10.0f, rect.h * 0.28f));
}

bool Context::clicked(const Rect& rect, bool enabled) const {
	return enabled && pointInRect(state_.mouseXWindow, state_.mouseYWindow, rect) && state_.mousePressed;
}

bool Context::held(const Rect& rect, bool enabled) const {
	return enabled && pointInRect(state_.mouseXWindow, state_.mouseYWindow, rect) && state_.mouseDown;
}

void Context::renderButton(const Rect& rect, const std::string& label, bool enabled) const {
	const bool hot = pointInRect(state_.mouseXWindow, state_.mouseYWindow, rect);
	const bool active = hot && state_.mouseDown;
	drawButton(rect, label, hot, active, enabled);
}

bool Context::button(const Rect& rect, const std::string& label, bool enabled) const {
	renderButton(rect, label, enabled);
	return clicked(rect, enabled);
}

bool Context::holdButton(const Rect& rect, const std::string& label, bool enabled) const {
	renderButton(rect, label, enabled);
	return held(rect, enabled);
}

} // namespace SimpleUI
