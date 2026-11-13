"""Comprehensive test suite for the AI Chatbot application.

Covers:
- Rule-based intent recognition and response engine (chatbot.py)
- Multi-turn context awareness (Explain more, Give an example, Why?, Compare them)
- Context isolation across different conversations
- Smart follow-up suggestions
- Flask web routes and REST API endpoints (app.py)

Run with:
    .\\.venv\\Scripts\\python.exe -m pytest -v
"""

from datetime import datetime
import pytest

from app import app
from chatbot import (
    clean_input,
    clear_conversation_context,
    detect_intent,
    get_bot_response,
    get_bot_response_details,
    get_conversation_context,
)


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
# 2. Context-Aware Multi-Turn Dialogue Tests
# ============================================================================


class TestContextAwareConversations:
    """Tests covering multi-turn follow-ups and smart suggestions."""

    def test_followup_without_context(self) -> None:
        """Follow-up questions in an empty session should provide friendly guidance."""
        followups = ["Explain more", "Give an example", "Why?", "Compare them"]
        for query in followups:
            details = get_bot_response_details(query, context=None)
            assert "don't have enough context yet" in details["response"].lower()
            assert isinstance(details["suggestions"], list)
            assert len(details["suggestions"]) >= 2

    def test_followup_explain_more_flow(self) -> None:
        """'Explain more' deepens the explanation for the previous topic."""
        # Turn 1: Ask about Machine Learning
        conv_id = "test_conv_explain_more"
        clear_conversation_context(conv_id)

        t1 = get_bot_response_details("What is Machine Learning?", conversation_id=conv_id)
        assert "Machine Learning" in t1["response"]
        assert t1["context"]["topic"] == "machine_learning"

        # Turn 2: Follow up with 'Explain more'
        t2 = get_bot_response_details("Explain more", conversation_id=conv_id)
        assert "Pipeline" in t2["response"] or "Training" in t2["response"]
        assert t2["context"]["topic"] == "machine_learning"
        assert len(t2["suggestions"]) >= 2

    def test_followup_give_example_flow(self) -> None:
        """'Give an example' provides a concrete example for the active topic/subtopic."""
        conv_id = "test_conv_example"
        clear_conversation_context(conv_id)

        # Turn 1: Ask about Python functions
        t1 = get_bot_response_details("What is a Python function?", conversation_id=conv_id)
        assert t1["context"]["topic"] == "programming"
        assert t1["context"]["subtopic"] == "functions"

        # Turn 2: Ask for an example
        t2 = get_bot_response_details("Give an example", conversation_id=conv_id)
        assert "def calculate_total" in t2["response"] or "def " in t2["response"]
        assert len(t2["suggestions"]) >= 2

    def test_followup_why_flow(self) -> None:
        """'Why?' explains importance and rationale behind the concept."""
        conv_id = "test_conv_why"
        clear_conversation_context(conv_id)

        # Turn 1: Ask about AI
        t1 = get_bot_response_details("What is Artificial Intelligence?", conversation_id=conv_id)
        assert t1["context"]["topic"] == "artificial_intelligence"

        # Turn 2: Ask Why
        t2 = get_bot_response_details("Why is that?", conversation_id=conv_id)
        assert "Why AI is Important" in t2["response"] or "Why" in t2["response"]
        assert len(t2["suggestions"]) >= 2

    def test_followup_compare_flow(self) -> None:
        """'Compare them' returns comparison based on topic in context."""
        conv_id = "test_conv_compare"
        clear_conversation_context(conv_id)

        # Turn 1: Ask about Supervised Learning
        t1 = get_bot_response_details("Explain supervised learning", conversation_id=conv_id)
        assert t1["context"]["topic"] == "machine_learning"

        # Turn 2: Ask to compare
        t2 = get_bot_response_details("Compare them", conversation_id=conv_id)
        assert "Supervised" in t2["response"] and "Unsupervised" in t2["response"]
        assert len(t2["suggestions"]) >= 2

    def test_context_isolation_between_conversations(self) -> None:
        """Verify context from Conversation A never leaks into Conversation B."""
        conv_a = "conversation_A_ml"
        conv_b = "conversation_B_prog"

        clear_conversation_context(conv_a)
        clear_conversation_context(conv_b)

        # Conversation A establishes Machine Learning context
        get_bot_response_details("What is Machine Learning?", conversation_id=conv_a)
        ctx_a = get_conversation_context(conv_a)
        assert ctx_a["topic"] == "machine_learning"

        # Conversation B establishes Python Loops context
        get_bot_response_details("Explain loops in Python", conversation_id=conv_b)
        ctx_b = get_conversation_context(conv_b)
        assert ctx_b["topic"] == "programming"
        assert ctx_b["subtopic"] == "loops"

        # Follow-up in Conversation A must remain in Machine Learning
        reply_a = get_bot_response_details("Give an example", conversation_id=conv_a)
        assert "Netflix" in reply_a["response"] or "recommendation" in reply_a["response"]
        assert "for lang in languages" not in reply_a["response"]

        # Follow-up in Conversation B must remain in Python Loops
        reply_b = get_bot_response_details("Give an example", conversation_id=conv_b)
        assert "for " in reply_b["response"]
        assert "Netflix" not in reply_b["response"]


# ============================================================================
# 3. Flask Application and API Tests (app.py)
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
        """Verify POST /api/chat responds with valid JSON, intent, suggestions, and timestamp."""
        payload = {"message": query}
        response = client.post("/api/chat", json=payload)

        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["message"] == query
        assert data["intent"] == expected_intent
        assert isinstance(data["response"], str)
        assert len(data["response"].strip()) > 0
        assert isinstance(data["suggestions"], list)
        assert len(data["suggestions"]) >= 2
        assert isinstance(data["context"], dict)
        # Check ISO timestamp format
        datetime.fromisoformat(data["timestamp"])

    def test_multi_turn_chat_api_with_conversation_id(self, client) -> None:
        """Test sending multi-turn follow-ups to /api/chat preserving conversation_id."""
        conv_id = "api_test_multi_turn"

        # Turn 1
        res1 = client.post("/api/chat", json={"message": "What is Python?", "conversation_id": conv_id})
        assert res1.status_code == 200
        data1 = res1.get_json()
        assert data1["context"]["topic"] == "programming"

        # Turn 2: 'Explain more'
        res2 = client.post("/api/chat", json={"message": "Explain more", "conversation_id": conv_id})
        assert res2.status_code == 200
        data2 = res2.get_json()
        assert "Python" in data2["response"] or "Standard library" in data2["response"].lower()
        assert data2["context"]["topic"] == "programming"

    def test_conversation_context_endpoints(self, client) -> None:
        """Test GET and DELETE on /api/context/<conversation_id>."""
        conv_id = "api_test_ctx_mgmt"

        # Establish context
        client.post("/api/chat", json={"message": "What is AI?", "conversation_id": conv_id})

        # GET context
        get_res = client.get(f"/api/context/{conv_id}")
        assert get_res.status_code == 200
        assert get_res.get_json()["context"]["topic"] == "artificial_intelligence"

        # DELETE context
        del_res = client.delete(f"/api/context/{conv_id}")
        assert del_res.status_code == 200
        assert del_res.get_json()["success"] is True

        # Verify context is cleared
        get_res_after = client.get(f"/api/context/{conv_id}")
        assert get_res_after.get_json()["context"]["topic"] is None

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
