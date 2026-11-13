"""Rule-based response engine for the AI Chatbot application.

This module implements intent recognition using regular expressions and
structured lookup tables to generate beginner-friendly responses without
requiring external APIs or paid dependencies.
"""

import re

# Ordered intent patterns: specific question intents are checked before
# generic conversational intents (such as greetings or thanks) so messages
# like "Hello, what is AI?" receive the topic answer directly.
INTENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "identity",
        re.compile(
            r"\b(?:who (?:are|made|created|built) you|what are you|"
            r"(?:what(?:'s| is)|tell me) your name|do you have a name|"
            r"about yourself|this (?:chat)?bot|"
            r"are you (?:a |an )?(?:bot|chatbot|robot|human|ai))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "capabilities",
        re.compile(
            r"\b(?:what can you do|what do you do|"
            r"what can you help(?: me)? with|how can you help(?: me)?|"
            r"your capabilities|your features|your skills|"
            r"what topics(?: do you know| can you (?:answer|cover|discuss))?|"
            r"what can i ask(?: you)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "machine_learning",
        re.compile(
            r"\b(?:machine learning|deep learning|supervised learning|"
            r"unsupervised learning|reinforcement learning|neural networks?|"
            r"decision trees?|random forests?|overfitting|underfitting|"
            r"train(?:ing)? (?:a )?models?|ml)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "artificial_intelligence",
        re.compile(
            r"\b(?:artificial intelligence|generative ai|narrow ai|general ai|"
            r"natural language processing|nlp|computer vision|turing test|"
            r"chatbots?|ai)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "programming",
        re.compile(
            r"\b(?:programming|coding|python|variables?|loops?|for loops?|"
            r"while loops?|functions?|conditionals?|if[- ]else|"
            r"(?:python |linked )?lists?|list vs tuple|tuples?|"
            r"dictionar(?:y|ies)|dicts?|arrays?|recursion|oop|"
            r"object[- ]oriented|debug(?:ging)?|syntax(?: errors?)?|"
            r"algorithms?|data structures?|regex|regular expressions?|"
            r"write code|source code|what is code)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "help",
        re.compile(
            r"^(?:please\s+)?help\b|"
            r"\b(?:help\s+please|please\s+help|need\s+help|help\s+me|"
            r"can\s+you\s+help|need\s+assistance|assist\s+me|"
            r"how\s+(?:do|can)\s+i\s+use\s+(?:this|you)|"
            r"instructions|user\s+guide)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "goodbye",
        re.compile(
            r"\b(?:bye(?:\s+bye)?|goodbye|good\s+bye|"
            r"see\s+(?:you|ya)(?:\s+later|\s+soon)?|take\s+care|"
            r"catch\s+you\s+later|farewell|"
            r"have\s+a\s+(?:good|nice|great)\s+day|exit|quit)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "thanks",
        re.compile(
            r"\b(?:thanks?(?:\s+a\s+lot|\s+so\s+much|\s+very\s+much)?|"
            r"thank\s+you(?:\s+so\s+much|\s+very\s+much)?|"
            r"thx|ty|appreciate\s+it|much\s+appreciated)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "greeting",
        re.compile(
            r"\b(?:hi|hello|hey|howdy|greetings|"
            r"good\s+(?:morning|afternoon|evening)|"
            r"what'?s\s+up|sup|how\s+are\s+you(?:\s+doing)?)\b",
            re.IGNORECASE,
        ),
    ),
]

STATIC_RESPONSES: dict[str, str] = {
    "empty": (
        "It looks like your message was empty. "
        "Please type a question or greeting so I can help!"
    ),
    "greeting": (
        "Hello! 👋 Welcome to the AI Chatbot. "
        "Ask me anything about Artificial Intelligence, Machine Learning, "
        "or basic programming concepts!"
    ),
    "identity": (
        "I am **AI Chatbot**, a rule-based conversational assistant built with "
        "Python and Streamlit for an internship project. "
        "I recognize your questions using Python functions and regular expressions—"
        "no paid APIs required!"
    ),
    "capabilities": (
        "Here is what I can help you with:\n"
        "- **Greetings & Casual Chat** (saying hello, thanks, or goodbye)\n"
        "- **Artificial Intelligence** (AI basics, NLP, computer vision, generative AI)\n"
        "- **Machine Learning** (ML basics, supervised/unsupervised/reinforcement learning, neural networks)\n"
        "- **Basic Programming** (Python, variables, loops, functions, conditionals, data structures, OOP, debugging)\n"
        "- **Help & Guidance** on how to use this chatbot"
    ),
    "thanks": (
        "You're very welcome! 😊 Let me know if you have any other questions "
        "about AI, Machine Learning, or programming."
    ),
    "goodbye": (
        "Goodbye! 👋 Thanks for chatting. Feel free to come back whenever "
        "you want to learn more!"
    ),
    "help": (
        "I'm here to help! Simply type a message in the chat box below and press Enter.\n\n"
        "**Example questions you can try:**\n"
        "- *Who are you?*\n"
        "- *What is Artificial Intelligence?*\n"
        "- *Explain supervised learning.*\n"
        "- *What is a Python function?*\n\n"
        "You can also click **Clear Chat** at the top to start a fresh conversation."
    ),
    "unknown": (
        "I'm not sure how to answer that yet—I'm a rule-based chatbot focused on "
        "**Artificial Intelligence**, **Machine Learning**, and **basic programming**.\n\n"
        "Try asking something like *\"What is AI?\"*, *\"Explain machine learning\"*, "
        "or *\"What is a loop in Python?\"*, or type **help** for more ideas."
    ),
}

# Sub-topic rules give more tailored answers within broad technical intents.
AI_SUBTOPICS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\b(?:natural language processing|nlp)\b", re.IGNORECASE),
        "**Natural Language Processing (NLP)** is a branch of AI that enables "
        "computers to read, understand, and generate human language. "
        "Common examples include chatbots, language translation, and sentiment analysis.",
    ),
    (
        re.compile(r"\bcomputer vision\b", re.IGNORECASE),
        "**Computer Vision** is an AI field focused on helping computers interpret "
        "images and videos—used in face recognition, medical imaging, and self-driving cars.",
    ),
    (
        re.compile(r"\bgenerative ai\b", re.IGNORECASE),
        "**Generative AI** refers to AI systems that create new content—such as "
        "text, images, audio, or code—by learning patterns from large datasets.",
    ),
    (
        re.compile(r"\bchatbots?\b", re.IGNORECASE),
        "A **chatbot** is a software application designed to simulate human conversation "
        "using either predefined rules (like me!) or advanced AI language models.",
    ),
]

DEFAULT_AI_RESPONSE: str = (
    "**Artificial Intelligence (AI)** is the field of computer science focused on "
    "building systems that can perform tasks usually requiring human intelligence—such as "
    "understanding language, recognizing images, reasoning, and making decisions.\n\n"
    "Most AI used today is **Narrow AI** (designed for specific tasks like chatbots or "
    "recommendations), whereas **General AI** (human-level reasoning across all tasks) "
    "remains a research goal."
)

ML_SUBTOPICS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"(?=.*\b(?:ai|artificial intelligence)\b)(?=.*\b(?:ml|machine learning)\b)",
            re.IGNORECASE,
        ),
        "**AI vs. Machine Learning:**\n"
        "- **Artificial Intelligence (AI)** is the broad goal of creating smart machines.\n"
        "- **Machine Learning (ML)** is a subset of AI where computers learn patterns "
        "from data instead of following only hand-written rules.",
    ),
    (
        re.compile(
            r"(?=.*\bsupervised\b)(?=.*\bunsupervised\b)",
            re.IGNORECASE,
        ),
        "**Supervised vs. Unsupervised Learning:**\n"
        "- **Supervised Learning** trains on *labeled* data (inputs paired with known answers), "
        "like predicting house prices or classifying spam.\n"
        "- **Unsupervised Learning** finds hidden patterns in *unlabeled* data, "
        "like grouping customers into segments (clustering).",
    ),
    (
        re.compile(r"\bunsupervised learning\b", re.IGNORECASE),
        "**Unsupervised Learning** is a type of Machine Learning that works with "
        "unlabeled data to discover hidden patterns or groups—such as customer clustering "
        "and anomaly detection.",
    ),
    (
        re.compile(r"\bsupervised learning\b", re.IGNORECASE),
        "**Supervised Learning** is a type of Machine Learning where a model learns from "
        "labeled examples (input-output pairs) to make predictions, such as spam detection "
        "(classification) or price prediction (regression).",
    ),
    (
        re.compile(r"\breinforcement learning\b", re.IGNORECASE),
        "**Reinforcement Learning** teaches an agent to make sequences of decisions by "
        "interacting with an environment and receiving rewards or penalties—commonly used "
        "in robotics and game-playing AI.",
    ),
    (
        re.compile(r"\b(?:deep learning|neural networks?)\b", re.IGNORECASE),
        "**Deep Learning** is a subfield of Machine Learning inspired by the human brain. "
        "It uses multi-layered **artificial neural networks** to automatically learn complex "
        "patterns in images, audio, and text.",
    ),
    (
        re.compile(r"\b(?:overfitting|underfitting)\b", re.IGNORECASE),
        "- **Overfitting** happens when a model memorizes training data too closely and "
        "performs poorly on new, unseen data.\n"
        "- **Underfitting** happens when a model is too simple to capture the underlying "
        "patterns in the data.",
    ),
]

DEFAULT_ML_RESPONSE: str = (
    "**Machine Learning (ML)** is a branch of AI where computers learn patterns from "
    "data and improve over time without being explicitly programmed for every scenario.\n\n"
    "The three main types are **Supervised Learning** (labeled data), "
    "**Unsupervised Learning** (unlabeled data), and **Reinforcement Learning** "
    "(learning via trial, error, and rewards)."
)

PROGRAMMING_SUBTOPICS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\bvariables?\b", re.IGNORECASE),
        "A **variable** is a named container that stores a value in memory so you can "
        "reuse and update it.\n\nExample in Python:\n```python\nname = \"Alice\"\nage = 21\n```",
    ),
    (
        re.compile(r"\b(?:for loops?|while loops?|loops?)\b", re.IGNORECASE),
        "A **loop** repeats a block of code multiple times:\n"
        "- A **`for` loop** iterates over a sequence (like a list or `range`).\n"
        "- A **`while` loop** repeats as long as a condition stays `True`.\n\n"
        "Example:\n```python\nfor i in range(3):\n    print(i)\n```",
    ),
    (
        re.compile(r"\bfunctions?\b", re.IGNORECASE),
        "A **function** is a reusable block of code that performs a specific task. "
        "In Python, functions are defined with `def`:\n\n"
        "```python\ndef greet(name):\n    return f\"Hello, {name}!\"\n```",
    ),
    (
        re.compile(r"\b(?:conditionals?|if[- ]else)\b", re.IGNORECASE),
        "**Conditionals** (`if`, `elif`, `else`) let your program make decisions "
        "based on whether a condition is `True` or `False`:\n\n"
        "```python\nif score >= 50:\n    print(\"Pass\")\nelse:\n    print(\"Try again\")\n```",
    ),
    (
        re.compile(
            r"\b(?:lists?|tuples?|dictionar(?:y|ies)|dicts?|arrays?|data structures?)\b",
            re.IGNORECASE,
        ),
        "Common **Python data structures** include:\n"
        "- **List** (`[1, 2, 3]`): ordered, mutable collection.\n"
        "- **Tuple** (`(1, 2, 3)`): ordered, immutable collection.\n"
        "- **Dictionary** (`{\"name\": \"Ada\"}`): stores key-value pairs for fast lookup.",
    ),
    (
        re.compile(r"\b(?:oop|object[- ]oriented)\b", re.IGNORECASE),
        "**Object-Oriented Programming (OOP)** organizes code around **classes** (blueprints) "
        "and **objects** (instances) that bundle data (attributes) and behavior (methods) together.",
    ),
    (
        re.compile(r"\b(?:debug(?:ging)?|syntax(?: errors?)?)\b", re.IGNORECASE),
        "**Debugging** is the process of finding and fixing errors (bugs) in code. "
        "**Syntax errors** occur when code breaks the language's grammar rules, while "
        "**logic errors** produce unexpected results even when the code runs.",
    ),
    (
        re.compile(r"\b(?:regex|regular expressions?)\b", re.IGNORECASE),
        "A **regular expression (regex)** is a sequence of characters that defines a "
        "search pattern for matching text—in Python, the built-in `re` module handles regex!",
    ),
    (
        re.compile(r"\bpython\b", re.IGNORECASE),
        "**Python** is a popular, beginner-friendly programming language known for its "
        "clean, readable syntax. It is widely used in web development, automation, "
        "data science, and AI/Machine Learning.",
    ),
]

DEFAULT_PROGRAMMING_RESPONSE: str = (
    "**Programming** is the process of writing clear instructions (code) that a computer "
    "can follow to solve a problem.\n\n"
    "Key building blocks include **variables** (storing data), **conditionals** (making decisions), "
    "**loops** (repeating steps), and **functions** (reusable code blocks)."
)

TOPIC_HANDLERS: dict[str, tuple[list[tuple[re.Pattern[str], str]], str]] = {
    "artificial_intelligence": (AI_SUBTOPICS, DEFAULT_AI_RESPONSE),
    "machine_learning": (ML_SUBTOPICS, DEFAULT_ML_RESPONSE),
    "programming": (PROGRAMMING_SUBTOPICS, DEFAULT_PROGRAMMING_RESPONSE),
}


def clean_input(user_message: str | None) -> str:
    """Safely normalize user input by trimming and collapsing whitespace."""
    if not isinstance(user_message, str):
        return ""
    return " ".join(user_message.strip().split())


def detect_intent(user_message: str | None) -> str:
    """Identify the primary intent name for a user message.

    Returns 'empty' for blank inputs, a matching intent name from
    INTENT_PATTERNS, or 'unknown' if no pattern matches.
    """
    cleaned_text = clean_input(user_message)
    if not cleaned_text:
        return "empty"

    for intent_name, pattern in INTENT_PATTERNS:
        if pattern.search(cleaned_text):
            return intent_name

    return "unknown"


def _match_topic_response(
    cleaned_text: str,
    topic_rules: list[tuple[re.Pattern[str], str]],
    default_response: str,
) -> str:
    """Return the first matching sub-topic response or fall back to default_response."""
    for pattern, response in topic_rules:
        if pattern.search(cleaned_text):
            return response
    return default_response


def get_bot_response(user_message: str | None) -> str:
    """Generate a rule-based chatbot response for the given user message."""
    cleaned_text = clean_input(user_message)
    intent = detect_intent(cleaned_text)

    if intent in TOPIC_HANDLERS:
        subtopics, default_response = TOPIC_HANDLERS[intent]
        return _match_topic_response(cleaned_text, subtopics, default_response)

    return STATIC_RESPONSES.get(intent, STATIC_RESPONSES["unknown"])

