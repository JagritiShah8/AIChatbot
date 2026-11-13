"""Flask Web Application for the Professional AI Chatbot.

Serves the standalone frontend and provides REST API endpoints for chat
interaction, conversation context management, health, and engine metadata.
"""

from datetime import datetime, timezone
import logging
import os
from flask import Flask, jsonify, render_template, request

from chatbot import (
    clean_input,
    clear_conversation_context,
    get_bot_response_details,
    get_conversation_context,
)

app = Flask(__name__, static_folder="static", template_folder="templates")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route("/")
def index():
    """Serve the main chatbot web interface."""
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Process a user message and return the chatbot response with context and suggestions."""
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

    conversation_id = data.get("conversation_id")
    if conversation_id is not None and not isinstance(conversation_id, str):
        conversation_id = str(conversation_id)

    client_context = data.get("context")
    if not isinstance(client_context, dict):
        client_context = None

    try:
        details = get_bot_response_details(
            cleaned_message,
            context=client_context,
            conversation_id=conversation_id,
        )

        return (
            jsonify({
                "success": True,
                "message": cleaned_message,
                "response": details["response"],
                "intent": details["intent"],
                "suggestions": details["suggestions"],
                "context": details["context"],
                "conversation_id": conversation_id,
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


@app.route("/api/context/<conversation_id>", methods=["GET", "DELETE"])
def conversation_context_endpoint(conversation_id: str):
    """Inspect or reset context for a specific conversation."""
    if request.method == "DELETE":
        clear_conversation_context(conversation_id)
        return jsonify({"success": True, "message": f"Context cleared for {conversation_id}"}), 200

    context = get_conversation_context(conversation_id)
    return jsonify({"success": True, "conversation_id": conversation_id, "context": context}), 200


@app.route("/api/health", methods=["GET"])
def health():
    """Health check and engine info endpoint."""
    return (
        jsonify({
            "status": "healthy",
            "service": "AI Chatbot Web Application",
            "engine": "Context-Aware Rule-Based Intent Recognition Engine",
            "version": "2.1.0",
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
