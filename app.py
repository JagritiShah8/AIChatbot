"""Flask Web Application for the Professional AI Chatbot.

Serves the standalone frontend and provides REST API endpoints for chat
interaction, conversation health, and engine metadata.
"""

from datetime import datetime, timezone
import logging
import os
from flask import Flask, jsonify, render_template, request

from chatbot import clean_input, detect_intent, get_bot_response

app = Flask(__name__, static_folder="static", template_folder="templates")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route("/")
def index():
    """Serve the main chatbot web interface."""
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Process a user message and return the chatbot response."""
    if not request.is_json:
        return (
            jsonify({"success": False, "error": "Request body must be valid JSON"}),
            400,
        )

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return (
            jsonify({"success": False, "error": "JSON payload must be an object"}),
            400,
        )

    raw_message = data.get("message")
    if raw_message is None:
        return (
            jsonify({
                "success": False,
                "error": "Missing 'message' field in request body",
            }),
            400,
        )

    if not isinstance(raw_message, str):
        return (
            jsonify({
                "success": False,
                "error": "'message' field must be a string",
            }),
            400,
        )

    cleaned_message = clean_input(raw_message)
    if not cleaned_message:
        return (
            jsonify({
                "success": False,
                "error": "Message cannot be empty or whitespace only",
            }),
            400,
        )

    try:
        intent = detect_intent(cleaned_message)
        response_text = get_bot_response(cleaned_message)

        return (
            jsonify({
                "success": True,
                "message": cleaned_message,
                "response": response_text,
                "intent": intent,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }),
            200,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Error generating chatbot response: %s", exc)
        return (
            jsonify({
                "success": False,
                "error": "An internal server error occurred while processing your message.",
            }),
            500,
        )


@app.route("/api/health", methods=["GET"])
def health():
    """Health check and engine info endpoint."""
    return (
        jsonify({
            "status": "healthy",
            "service": "AI Chatbot Web Application",
            "engine": "Rule-Based Intent Recognition Engine",
            "version": "2.0.0",
        }),
        200,
    )


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors for both API and page requests."""
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "API endpoint not found"}), 404
    return render_template("index.html"), 404


@app.errorhandler(500)
def server_error(e):
    """Handle internal server errors gracefully."""
    if request.path.startswith("/api/"):
        return (
            jsonify({
                "success": False,
                "error": "Internal server error. Please try again later.",
            }),
            500,
        )
    return "Internal Server Error", 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=True)
