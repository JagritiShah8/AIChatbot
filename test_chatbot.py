"""Comprehensive test suite for the AI Chatbot application.

Covers the rule-based response engine (chatbot.py) and the Flask web/API layer (app.py).
Run with:
    .\\.venv\\Scripts\\python.exe -m pytest -v
"""

from datetime import datetime
import pytest

from app import app
from chatbot import clean_input, detect_intent, get_bot_response


# ============================================================================
# 1. Response Engine Tests (chatbot.py)
# ============================================================================


class TestChatbotEngine:
    """Tests covering intent classification and response generation in chatbot.py."""

    @pytest.mark.parametrize(
        "user_input, expected_intent",
        [
            ("Hello there!", "greeting"),
            ("Good morning, how are you?", "greeting"),
            ("Who are you?", "identity"),
            ("Tell me your name", "identity"),
            ("What can you do?", "capabilities"),
            ("What are your features?", "capabilities"),
            ("Thanks a lot!", "thanks"),
            ("Thank you so much", "thanks"),
            ("Goodbye, see you later!", "goodbye"),
            ("Bye bye!", "goodbye"),
            ("What is Artificial Intelligence?", "artificial_intelligence"),
            ("Explain natural language processing", "artificial_intelligence"),
            ("Tell me about computer vision", "artificial_intelligence"),
            ("What is generative AI?", "artificial_intelligence"),
            ("What is Machine Learning?", "machine_learning"),
            ("Explain supervised learning", "machine_learning"),
            ("Supervised vs unsupervised learning", "machine_learning"),
            ("What are neural networks?", "machine_learning"),
            ("What is a Python function?", "programming"),
            ("Explain loops in Python", "programming"),
            ("What are variables?", "programming"),
            ("What is object-oriented programming?", "programming"),
            ("I need help", "help"),
            ("How do I use this?", "help"),
            ("What is the stock price of Apple?", "unknown"),
            ("Can you order a pizza for me?", "unknown"),
        ],
    )
    def test_intent_detection(self, user_input: str, expected_intent: str) -> None:
        """Verify intent detection across all supported categories and unknown queries."""
        detected = detect_intent(user_input)
        assert (
            detected == expected_intent
        ), f"Expected '{expected_intent}' for '{user_input}', got '{detected}'"

        response = get_bot_response(user_input)
        assert isinstance(response, str)
        assert len(response.strip()) > 10

    @pytest.mark.parametrize(
        "empty_input",
        ["", "   ", "\t\n  ", None],
    )
    def test_empty_and_invalid_inputs(self, empty_input: str | None) -> None:
        """Verify empty, whitespace-only, and None inputs are safely handled."""
        assert detect_intent(empty_input) == "empty"
        response = get_bot_response(empty_input)
        assert isinstance(response, str)
        assert "empty" in response.lower()

    def test_clean_input_helper(self) -> None:
        """Ensure clean_input trims and normalizes multiple spaces."""
        assert clean_input("   hello    world   ") == "hello world"
        assert clean_input("") == ""
        assert clean_input(None) == ""


# ============================================================================
# 2. Flask Application and API Tests (app.py)
# ============================================================================


@pytest.fixture
def client():
    """Create a Flask test client instance."""
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


class TestFlaskRoutes:
    """Integration tests for web routes and REST API endpoints."""

    def test_homepage_loads(self, client) -> None:
        """Ensure the frontend HTML page loads successfully."""
        response = client.get("/")
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert "AI Chatbot" in html
        assert "Rule-Based" in html
        assert "New Conversation" in html

    def test_health_endpoint(self, client) -> None:
        """Verify GET /api/health returns valid service metadata."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"
        assert "Rule-Based" in data["engine"]

    @pytest.mark.parametrize(
        "query, expected_intent",
        [
            ("Hello", "greeting"),
            ("What is AI?", "artificial_intelligence"),
            ("What is machine learning?", "machine_learning"),
            ("What is a function in Python?", "programming"),
            ("Who created you?", "identity"),
            ("Help me please", "help"),
            ("Goodbye", "goodbye"),
        ],
    )
    def test_successful_chat_api(
        self, client, query: str, expected_intent: str
    ) -> None:
        """Verify POST /api/chat responds with valid JSON, intent, and message."""
        payload = {"message": query}
        response = client.post("/api/chat", json=payload)

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["message"] == query
        assert data["intent"] == expected_intent
        assert isinstance(data["response"], str)
        assert len(data["response"].strip()) > 0
        # Check ISO timestamp format
        datetime.fromisoformat(data["timestamp"])

    def test_invalid_api_missing_json(self, client) -> None:
        """POST /api/chat without JSON body returns 400 Bad Request."""
        response = client.post(
            "/api/chat",
            data="not json",
            content_type="text/plain",
        )
        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False
        assert "valid JSON" in data["error"]

    def test_invalid_api_missing_message_field(self, client) -> None:
        """POST /api/chat with missing 'message' field returns 400."""
        response = client.post("/api/chat", json={"other_key": "value"})
        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False
        assert "Missing 'message' field" in data["error"]

    def test_invalid_api_non_string_message(self, client) -> None:
        """POST /api/chat with non-string message returns 400."""
        response = client.post("/api/chat", json={"message": 12345})
        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False
        assert "must be a string" in data["error"]

    def test_invalid_api_empty_message(self, client) -> None:
        """POST /api/chat with empty or whitespace message returns 400."""
        for empty_text in ["", "   ", "\t\n  "]:
            response = client.post("/api/chat", json={"message": empty_text})
            assert response.status_code == 400
            data = response.get_json()
            assert data["success"] is False
            assert "empty" in data["error"].lower()

    def test_api_404_error_handler(self, client) -> None:
        """Non-existent /api/ endpoint returns JSON 404."""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404
        data = response.get_json()
        assert data["success"] is False
        assert "not found" in data["error"].lower()
