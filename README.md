# 🤖 AI Chatbot — Interactive Rule-Based Web Application

A professional, standalone AI Chatbot website built with **Python Flask**, **Vanilla JavaScript**, **CSS3**, and **HTML5**. It is powered by a modular **context-aware rule-based intent recognition engine** using regular expressions, multi-turn state tracking, and domain knowledge dictionaries—requiring **zero paid APIs, API keys, or external service dependencies**.

---

## ✨ Features

### 🧠 Context-Aware Multi-Turn Conversations
- **Follow-Up Understanding**: Understands natural conversational follow-ups such as:
  - *"Explain more"* / *"Can you elaborate?"*
  - *"Give an example"* / *"Show me a code example"*
  - *"Why?"* / *"Why is that important?"*
  - *"Compare them"* / *"What is the difference?"*
- **Per-Conversation Context Isolation**: Context is maintained separately for every chat session (both in-memory via `conversation_id` and client-side via `localStorage`), ensuring discussions in one tab never bleed into another.
- **Friendly No-Context Guidance**: Asking follow-ups in a fresh chat politely guides the user to ask an introductory question first.

### 💡 Smart Dynamic Follow-Up Suggestions
- **Clickable Prompt Pills**: Every assistant message displays 2–3 contextual suggestion chips tailored to the current topic.
- **Instant Dispatch**: Clicking any suggestion chip immediately sends it as a user message.

### ⚡ Interactive Message Controls
- **Copy Response**: One-click copy with smooth icon and label feedback ("Copied!") and subtle toast notifications.
- **Regenerate Response**: Re-processes the preceding prompt with current context and replaces the response in place.
- **Feedback Rating**: Thumbs-up (👍) and Thumbs-down (👎) buttons with active state highlighting and subtle confirmation toasts.

### 🎨 Modern & Polished Interface
- **Dark Theme Design System**: Typography using `Plus Jakarta Sans` and `JetBrains Mono`, subtle borders, glassmorphism, and smooth animations.
- **Multi-Conversation Management**: Create new chats, switch between chats, delete individual chats, and clear current conversation.
- **Browser Persistence**: Complete conversation history, messages, context, and feedback ratings persist across page refreshes via `localStorage`.
- **Typing Indicator & Timestamps**: Animated dot wave while responses are generated, plus per-message timestamps.
- **Keyboard Shortcuts**: <kbd>Enter</kbd> to send, <kbd>Shift</kbd> + <kbd>Enter</kbd> for a new line, and <kbd>Esc</kbd> to dismiss modals or close the mobile sidebar drawer.
- **XSS-Safe Markdown Support**: Safely formats `<pre><code>` code blocks, inline code, bold text, and bullet lists without unsafe HTML injection.
- **REST API Backend**: Modular Flask API (`POST /api/chat`, `GET /api/health`, `GET/DELETE /api/context/<id>`) with JSON validation and graceful error handling.

---

## 🏗️ Architecture

```
AIChatbot/
├── app.py                 # Flask web server & REST API endpoints (/api/chat, /api/health, /api/context)
├── chatbot.py             # Context-aware rule-based engine, multi-turn memory & suggestion system
├── streamlit_app.py       # Preserved original Streamlit chat application
├── test_chatbot.py        # Automated test suite (53 unit & integration tests)
├── requirements.txt       # Project dependencies
├── .gitignore             # Git ignore configuration
├── templates/
│   └── index.html         # Accessible, semantic HTML5 chat interface with toast container
└── static/
    ├── css/
    │   └── style.css      # Modern dark-mode responsive design system with action bars & suggestion pills
    └── js/
        └── chat.js        # Multi-turn context management, message actions, suggestions, & localStorage
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+ (tested with Python 3.14.2)
- Virtual environment in `.venv`

### 2. Installation

Install dependencies inside the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Launching the Web Application

Run the Flask application:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🧪 Running the Tests

Execute the comprehensive pytest test suite inside the virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

All 53 test cases verify:
- Intent detection across all categories (greetings, identity, capabilities, AI, ML, programming, help, goodbye, thanks)
- Subtopic responses (NLP, computer vision, generative AI, supervised/unsupervised ML, functions, loops, variables, OOP)
- Context-aware follow-ups (`Explain more`, `Give an example`, `Why?`, `Compare them`)
- Context isolation across independent conversations (no data leakage)
- Empty, whitespace, and `None` input handling
- Flask homepage (`GET /`) and health check (`GET /api/health`)
- Chat API (`POST /api/chat`) response structure, suggestions, context, and timestamp validation
- Context management endpoints (`GET /api/context/<id>`, `DELETE /api/context/<id>`)
- Missing JSON, missing keys, invalid types, and empty strings (400 Bad Request)
- 404 error handler for unknown API routes

---

## 🔌 API Reference

### `POST /api/chat`
Processes a message with context support and returns the intent, response, and smart suggestions.

**Request:**
```http
POST /api/chat HTTP/1.1
Content-Type: application/json

{
  "message": "Give an example",
  "conversation_id": "conv_1700000000_abcde",
  "context": {
    "topic": "machine_learning",
    "subtopic": "supervised_learning"
  }
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Give an example",
  "intent": "followup_example",
  "response": "**Supervised Learning Example:** Medical diagnosis models...",
  "suggestions": [
    "What is Unsupervised Learning?",
    "Explain Overfitting",
    "Why is Supervised Learning used?"
  ],
  "context": {
    "topic": "machine_learning",
    "subtopic": "supervised_learning",
    "intent": "followup_example"
  },
  "conversation_id": "conv_1700000000_abcde",
  "timestamp": "2026-11-13T06:20:12.123456+00:00"
}
```

### `GET /api/context/<conversation_id>`
Inspects current active context for a conversation.

### `DELETE /api/context/<conversation_id>`
Resets in-memory context for a conversation.

### `GET /api/health`
Returns service status and engine metadata.

---

## 📱 Legacy Streamlit Interface

The original Streamlit interface is preserved in `streamlit_app.py` and can be run at any time via:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```
