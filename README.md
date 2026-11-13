# 🤖 AI Chatbot — Standalone Web Application

A professional, standalone AI Chatbot website built with **Python Flask**, **Vanilla JavaScript**, **CSS3**, and **HTML5**. It is powered by a modular **rule-based intent recognition engine** using regular expressions and domain knowledge dictionaries—requiring **zero paid APIs, API keys, or external service dependencies**.

---

## ✨ Features

- **Modern & Polished Interface**: Inspired by modern AI chat applications with a sleek dark theme, refined typography (`Plus Jakarta Sans` & `JetBrains Mono`), smooth animations, and responsive layouts for desktop, tablet, and mobile.
- **Rule-Based Conversational Engine**: Accurate intent recognition for greetings, bot identity, capabilities, artificial intelligence, machine learning, programming concepts, help, thanks, goodbye, and fallback handling.
- **Multiple Conversations**: Create new chats, switch between conversations, and delete chats with auto-generated titles derived from the first user prompt.
- **Browser Persistence**: Conversations are saved locally in `localStorage`, maintaining complete chat history across page refreshes.
- **Interactive Prompts**: One-click suggested prompt cards for instant answers on common AI/ML/Python topics.
- **Typing Indicator & Timestamps**: Visual feedback with animated typing dots while responses are returned, along with individual message timestamps.
- **Keyboard Shortcuts**: <kbd>Enter</kbd> to send, <kbd>Shift</kbd> + <kbd>Enter</kbd> for a new line, and <kbd>Esc</kbd> to dismiss modals or close the mobile sidebar drawer.
- **XSS-Safe Markdown Support**: Safely parses code blocks, inline code, bold text, and bullet lists without unsafe HTML injection.
- **REST API Backend**: Modular Flask API (`POST /api/chat`, `GET /api/health`) with JSON validation and graceful error handling.
- **Automated Test Suite**: 45 passing unit and integration tests covering intent detection, edge cases, routes, and error handling.

---

## 🏗️ Architecture

```
AIChatbot/
├── app.py                 # Flask web server & REST API endpoints (/api/chat, /api/health)
├── chatbot.py             # Modular rule-based intent recognition & response engine
├── streamlit_app.py       # Preserved original Streamlit chat application
├── test_chatbot.py        # Complete automated test suite using pytest
├── requirements.txt       # Project dependencies
├── .gitignore             # Git ignore configuration
├── templates/
│   └── index.html         # Accessible, semantic HTML5 chat interface
└── static/
    ├── css/
    │   └── style.css      # Modern dark-mode responsive design system
    └── js/
        └── chat.js        # Multi-conversation management & API integration
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+ (tested with Python 3.14.2)
- Virtual environment in `.venv`

### 2. Installation

Activate the existing virtual environment and install dependencies:

```powershell
# Using the project's virtual environment
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Launching the Web Application

Run the Flask application:

```powershell
.\.venv\Scripts\python.exe app.py
```

Once started, open your web browser and navigate to:
```
http://127.0.0.1:5000
```

*(Optional: To run on a different port, set the `PORT` environment variable before starting: `$env:PORT="8080"; .\.venv\Scripts\python.exe app.py`)*

---

## 🧪 Running the Tests

Execute the comprehensive pytest test suite inside the virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

All 45 test cases verify:
- Intent detection across all categories (greetings, identity, capabilities, AI, ML, programming, help, goodbye, thanks)
- Subtopic responses (NLP, computer vision, generative AI, supervised/unsupervised ML, functions, loops, variables, OOP)
- Empty, whitespace, and `None` input handling
- Flask homepage (`GET /`) and health check (`GET /api/health`)
- Chat API (`POST /api/chat`) response structure and timestamp validation
- Missing JSON, missing keys, invalid types, and empty strings (400 Bad Request)
- 404 error handler for unknown API routes

---

## 🔌 API Reference

### `POST /api/chat`
Processes a message and returns the intent and rule-based response.

**Request:**
```http
POST /api/chat HTTP/1.1
Content-Type: application/json

{
  "message": "What is machine learning?"
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "What is machine learning?",
  "intent": "machine_learning",
  "response": "**Machine Learning (ML)** is a branch of AI where computers learn patterns from data...",
  "timestamp": "2026-11-13T00:06:20.910621+00:00"
}
```

**Error Response (400 Bad Request):**
```json
{
  "success": false,
  "error": "Message cannot be empty or whitespace only"
}
```

### `GET /api/health`
Returns service status and engine metadata.

---

## 📱 Legacy Streamlit Interface

The original Streamlit interface is preserved in `streamlit_app.py` and can be run at any time via:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

