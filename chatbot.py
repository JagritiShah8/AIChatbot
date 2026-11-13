"""Rule-based response engine for the AI Chatbot application.

This module implements:
- Intent recognition using regular expressions and lookup tables
- Context-aware multi-turn dialogues (follow-ups like 'Explain more', 'Give an example', 'Why?', 'Compare them')
- Context management per conversation to prevent leakage across chats
- Smart dynamic follow-up suggestions for all responses
- Safe input normalization and backward-compatible helper functions
"""

from typing import Any
import re

# ============================================================================
# 1. Regex Intent Patterns
# ============================================================================

# Context follow-up patterns
FOLLOWUP_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "explain_more",
        re.compile(
            r"^(?:(?:can you |please )?(?:explain|tell me|elaborate)(?: more| further)?|more details?|go deeper|more info|explain further)\??$|"
            r"\b(?:explain more|tell me more|can you elaborate|elaborate|more details?|go deeper|more info|explain further)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "example",
        re.compile(
            r"^(?:(?:give|show|provide)(?: me)? (?:an? )?(?:example|sample|code example)|example please|can you give an example|real[- ]world example)\??$|"
            r"\b(?:give (?:an? )?example|show (?:an? )?example|example please|code example|real[- ]world example|provide (?:an? )?example|can you give an example|give me an example)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "why",
        re.compile(
            r"^(?:why\??|why so\??|why is that\??|why is it (?:important|needed|used)\??|"
            r"why do we need (?:this|it)\??|why does this matter\??)$|"
            r"\b(?:why is that|why is it important|why do we need this|why use it|why does this matter)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "compare",
        re.compile(
            r"^(?:compare them|compare|contrast them)\??$|"
            r"\b(?:compare them|compare|what is the difference|how do they compare|contrast them|difference between them|compare both)\b",
            re.IGNORECASE,
        ),
    ),
]

# Standard topic and conversational intent patterns
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

# Subtopic detection patterns
SUBTOPIC_PATTERNS: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    "artificial_intelligence": [
        ("nlp", re.compile(r"\b(?:natural language processing|nlp)\b", re.IGNORECASE)),
        ("computer_vision", re.compile(r"\bcomputer vision\b", re.IGNORECASE)),
        ("generative_ai", re.compile(r"\bgenerative ai\b", re.IGNORECASE)),
        ("chatbots", re.compile(r"\bchatbots?\b", re.IGNORECASE)),
    ],
    "machine_learning": [
        ("supervised_vs_unsupervised", re.compile(r"(?=.*\bsupervised\b)(?=.*\bunsupervised\b)", re.IGNORECASE)),
        ("supervised_learning", re.compile(r"\bsupervised learning\b", re.IGNORECASE)),
        ("unsupervised_learning", re.compile(r"\bunsupervised learning\b", re.IGNORECASE)),
        ("reinforcement_learning", re.compile(r"\breinforcement learning\b", re.IGNORECASE)),
        ("deep_learning", re.compile(r"\b(?:deep learning|neural networks?)\b", re.IGNORECASE)),
        ("overfitting", re.compile(r"\b(?:overfitting|underfitting)\b", re.IGNORECASE)),
    ],
    "programming": [
        ("variables", re.compile(r"\bvariables?\b", re.IGNORECASE)),
        ("loops", re.compile(r"\b(?:for loops?|while loops?|loops?)\b", re.IGNORECASE)),
        ("functions", re.compile(r"\bfunctions?\b", re.IGNORECASE)),
        ("conditionals", re.compile(r"\b(?:conditionals?|if[- ]else)\b", re.IGNORECASE)),
        ("data_structures", re.compile(r"\b(?:lists?|tuples?|dictionar(?:y|ies)|dicts?|arrays?|data structures?)\b", re.IGNORECASE)),
        ("oop", re.compile(r"\b(?:oop|object[- ]oriented)\b", re.IGNORECASE)),
        ("debugging", re.compile(r"\b(?:debug(?:ging)?|syntax(?: errors?)?)\b", re.IGNORECASE)),
        ("regex", re.compile(r"\b(?:regex|regular expressions?)\b", re.IGNORECASE)),
        ("python", re.compile(r"\bpython\b", re.IGNORECASE)),
    ],
}

# ============================================================================
# 2. Domain Knowledge Base (Topic Answers, Follow-ups, and Suggestions)
# ============================================================================

KNOWLEDGE_BASE: dict[str, Any] = {
    "artificial_intelligence": {
        "default": (
            "**Artificial Intelligence (AI)** is the field of computer science focused on "
            "building systems that can perform tasks usually requiring human intelligence—such as "
            "understanding language, recognizing images, reasoning, and making decisions.\n\n"
            "Most AI used today is **Narrow AI** (designed for specific tasks like chatbots or "
            "recommendations), whereas **General AI** (human-level reasoning across all tasks) "
            "remains a research goal."
        ),
        "explain_more": (
            "**How Modern AI Works:**\n"
            "1. **Perception**: Collecting data from the environment via text, sensors, audio, or cameras.\n"
            "2. **Representation & Reasoning**: Mapping inputs into mathematical representations (vectors and embeddings).\n"
            "3. **Learning & Optimization**: Using algorithms to minimize error rates and improve performance.\n"
            "4. **Action**: Producing an answer, recommendation, or robotic movement."
        ),
        "example": (
            "**Real-World Example of AI:**\n"
            "Autonomous self-driving vehicles (such as Waymo). They combine **computer vision** "
            "to recognize pedestrians and street signs, sensor fusion (LiDAR + Radar), and "
            "real-time path planning algorithms to navigate safely."
        ),
        "why": (
            "**Why AI is Important:**\n"
            "AI automates repetitive and complex cognitive tasks at massive scale, detects non-linear "
            "patterns in vast datasets that humans cannot spot, and accelerates breakthroughs in healthcare, "
            "climate modeling, and engineering."
        ),
        "compare": (
            "**Narrow AI vs. General AI (AGI):**\n"
            "- **Narrow AI (ANI)**: Highly specialized for one domain (e.g., chess, FaceID, spam filters). It exists today.\n"
            "- **General AI (AGI)**: Hypothetical system capable of learning and solving any intellectual task a human can."
        ),
        "suggestions": [
            "Give an example of AI",
            "What is Machine Learning?",
            "Narrow vs General AI",
        ],
        "subtopics": {
            "nlp": {
                "default": (
                    "**Natural Language Processing (NLP)** is a branch of AI that enables "
                    "computers to read, understand, and generate human language. "
                    "Common examples include chatbots, language translation, and sentiment analysis."
                ),
                "explain_more": (
                    "**Key NLP Steps:**\n"
                    "- **Tokenization**: Splitting text into words or subwords.\n"
                    "- **Embeddings**: Converting words into high-dimensional numerical vectors where similar words cluster together.\n"
                    "- **Transformers**: Neural architectures using self-attention to process entire sentences simultaneously."
                ),
                "example": (
                    "**NLP Example:** Email spam filters and predictive text (Smart Compose) that analyze incoming "
                    "words and predict either the email's label or the next most probable word."
                ),
                "why": (
                    "**Why NLP Matters:** Most human knowledge is stored in unstructured language. NLP allows computers "
                    "to search, translate, and extract insights from millions of documents in seconds."
                ),
                "compare": (
                    "**NLP vs. Computer Vision:** NLP focuses on sequential linguistic tokens (words/audio), whereas "
                    "Computer Vision interprets 2D/3D pixel matrices from images and video."
                ),
                "suggestions": [
                    "Give an NLP example",
                    "What is Computer Vision?",
                    "How do chatbots work?",
                ],
            },
            "computer_vision": {
                "default": (
                    "**Computer Vision** is an AI field focused on helping computers interpret "
                    "images and videos—used in face recognition, medical imaging, and self-driving cars."
                ),
                "explain_more": (
                    "**How Computer Vision Works:**\n"
                    "Images are treated as numerical pixel matrices. Convolutional Neural Networks (CNNs) slide filters "
                    "over pixels to detect low-level edges, then textures, shapes, and finally full objects."
                ),
                "example": (
                    "**Computer Vision Example:** Smartphone **Face Unlock**, which maps depth points on your face "
                    "and matches geometric ratios against stored biometric templates."
                ),
                "why": (
                    "**Why Computer Vision Matters:** Vision is humanity's highest-bandwidth sense. Enabling machines "
                    "to see powers automated radiology scans, manufacturing inspection, and robotics."
                ),
                "compare": (
                    "**Image Classification vs. Object Detection:** Classification asks *'What is in this picture?'*, "
                    "while Object Detection asks *'Where is it?'* and draws bounding boxes around multiple items."
                ),
                "suggestions": [
                    "Give a Computer Vision example",
                    "What is NLP?",
                    "What are neural networks?",
                ],
            },
            "generative_ai": {
                "default": (
                    "**Generative AI** refers to AI systems that create new content—such as "
                    "text, images, audio, or code—by learning patterns from large datasets."
                ),
                "explain_more": (
                    "**How Generative AI Works:**\n"
                    "Models are trained on massive corpora to learn probability distributions. When prompted, they "
                    "sample from those distributions to generate statistically coherent, novel outputs."
                ),
                "example": (
                    "**Generative AI Example:** Large language models drafting essays and code, or diffusion models "
                    "creating photorealistic art from descriptive text prompts."
                ),
                "why": (
                    "**Why Generative AI Matters:** It shifts AI from purely analytical classification to creative "
                    "assistance, speeding up coding, writing, and design workflows."
                ),
                "compare": (
                    "**Generative vs. Discriminative Models:** Discriminative models classify inputs (e.g. *'Is this photo a cat?'*), "
                    "while Generative models create new instances (e.g. *'Draw a photo of a cat'*)."
                ),
                "suggestions": [
                    "Give a Generative AI example",
                    "What is Machine Learning?",
                    "What is Deep Learning?",
                ],
            },
            "chatbots": {
                "default": (
                    "A **chatbot** is a software application designed to simulate human conversation "
                    "using either predefined rules (like me!) or advanced AI language models."
                ),
                "explain_more": (
                    "**Types of Chatbots:**\n"
                    "- **Rule-Based**: Deterministic pattern matching with high speed, predictable answers, and zero hallucination.\n"
                    "- **LLM-Powered**: Deep neural models with fluid language capabilities that require high computational power."
                ),
                "example": (
                    "**Chatbot Example:** Customer support bots on banking websites that recognize account balance queries "
                    "and guide customers step-by-step through transaction history."
                ),
                "why": (
                    "**Why Chatbots Matter:** They provide 24/7 immediate assistance, handle repetitive customer queries, "
                    "and eliminate long support hold times."
                ),
                "compare": (
                    "**Rule-Based vs. LLM Chatbots:** Rule-based bots follow explicit logic and regular expressions without "
                    "external dependencies. LLM bots generate probabilistic text but require strict safety guardrails."
                ),
                "suggestions": [
                    "What is NLP?",
                    "What can you do?",
                    "What is AI?",
                ],
            },
        },
    },
    "machine_learning": {
        "default": (
            "**Machine Learning (ML)** is a branch of AI where computers learn patterns from "
            "data and improve over time without being explicitly programmed for every scenario.\n\n"
            "The three main types are **Supervised Learning** (labeled data), "
            "**Unsupervised Learning** (unlabeled data), and **Reinforcement Learning** "
            "(learning via trial, error, and rewards)."
        ),
        "explain_more": (
            "**The Machine Learning Pipeline:**\n"
            "1. **Data Prep**: Cleaning data and splitting into Training and Test sets.\n"
            "2. **Training**: Feeding features into an algorithm to calculate mathematical weights.\n"
            "3. **Loss Function**: Measuring how far predictions deviate from actual values.\n"
            "4. **Evaluation**: Testing the model on unseen data to verify accuracy and prevent overfitting."
        ),
        "example": (
            "**Real-World Example of ML:**\n"
            "Streaming recommendation systems (Spotify / Netflix). The algorithm analyzes your past listening/viewing history, "
            "finds users with similar behavioral vectors, and recommends new titles."
        ),
        "why": (
            "**Why Machine Learning Matters:**\n"
            "Traditional software requires humans to manually write rules for every edge case. When problems become too complex "
            "(like fraud detection or speech recognition), ML lets the computer discover optimal rules directly from historical data."
        ),
        "compare": (
            "**AI vs. Machine Learning:**\n"
            "- **Artificial Intelligence (AI)** is the broad vision of creating smart systems.\n"
            "- **Machine Learning (ML)** is the specific practical subset of AI that uses data to learn patterns."
        ),
        "suggestions": [
            "Supervised vs Unsupervised ML",
            "Give an ML example",
            "What is Deep Learning?",
        ],
        "subtopics": {
            "supervised_learning": {
                "default": (
                    "**Supervised Learning** is a type of Machine Learning where a model learns from "
                    "labeled examples (input-output pairs) to make predictions, such as spam detection "
                    "(classification) or price prediction (regression)."
                ),
                "explain_more": (
                    "**Subcategories of Supervised Learning:**\n"
                    "- **Classification**: Predicting discrete categories (e.g. *Spam vs. Not Spam*, *Cat vs. Dog*).\n"
                    "- **Regression**: Predicting continuous numeric values (e.g. *House price*, *Stock price*, *Temperature*)."
                ),
                "example": (
                    "**Supervised Learning Example:** Medical diagnosis models trained on thousands of labeled patient scans "
                    "to predict whether a scan indicates disease or healthy tissue."
                ),
                "why": (
                    "**Why Supervised Learning:** Because labeled ground truth provides direct feedback to minimize loss, "
                    "making it the most accurate and widely adopted commercial ML approach."
                ),
                "compare": (
                    "**Supervised vs. Unsupervised Learning:** Supervised learning requires human-labeled training data with "
                    "known target answers. Unsupervised learning discovers hidden clusters in raw data without labels."
                ),
                "suggestions": [
                    "Give an example of Supervised Learning",
                    "What is Unsupervised Learning?",
                    "Explain Overfitting",
                ],
            },
            "unsupervised_learning": {
                "default": (
                    "**Unsupervised Learning** is a type of Machine Learning that works with "
                    "unlabeled data to discover hidden patterns or groups—such as customer clustering "
                    "and anomaly detection."
                ),
                "explain_more": (
                    "**Key Unsupervised Techniques:**\n"
                    "- **Clustering (e.g., K-Means)**: Grouping similar items together.\n"
                    "- **Dimensionality Reduction (e.g., PCA)**: Compressing hundreds of features into key signals.\n"
                    "- **Association Rules**: Discovering relationships like *'People who buy bread also buy butter'*."
                ),
                "example": (
                    "**Unsupervised Learning Example:** Customer segmentation in e-commerce, where an algorithm groups "
                    "millions of shoppers into distinct buyer personas based solely on browsing patterns."
                ),
                "why": (
                    "**Why Unsupervised Learning:** Most real-world data lacks labels because labeling is slow and expensive. "
                    "Unsupervised learning discovers insights without human annotation."
                ),
                "compare": (
                    "**Clustering vs. Classification:** Classification assigns inputs to predefined labels learned from examples. "
                    "Clustering groups data points purely by geometric similarity without knowing what the groups represent."
                ),
                "suggestions": [
                    "What is Supervised Learning?",
                    "Give an Unsupervised ML example",
                    "What is Reinforcement Learning?",
                ],
            },
            "reinforcement_learning": {
                "default": (
                    "**Reinforcement Learning** teaches an agent to make sequences of decisions by "
                    "interacting with an environment and receiving rewards or penalties—commonly used "
                    "in robotics and game-playing AI."
                ),
                "explain_more": (
                    "**Reinforcement Learning Components:**\n"
                    "- **Agent**: The learner or decision maker.\n"
                    "- **Environment**: The world the agent interacts with.\n"
                    "- **Action & State**: The choices available and current situation.\n"
                    "- **Reward**: Positive or negative feedback guiding policy optimization."
                ),
                "example": (
                    "**Reinforcement Learning Example:** AlphaGo and autonomous robotic arms learning how to grasp "
                    "irregularly shaped items through trial and error over millions of simulation steps."
                ),
                "why": (
                    "**Why Reinforcement Learning:** It is ideal for sequential decision-making where there is no static "
                    "labeled dataset, only an active environment with delayed consequences."
                ),
                "compare": (
                    "**Reinforcement vs. Supervised Learning:** Supervised learning receives explicit correct answers from a teacher. "
                    "Reinforcement learning learns independently through environmental trial, error, and reward signals."
                ),
                "suggestions": [
                    "Give an RL example",
                    "What is Supervised Learning?",
                    "What are Neural Networks?",
                ],
            },
            "deep_learning": {
                "default": (
                    "**Deep Learning** is a subfield of Machine Learning inspired by the human brain. "
                    "It uses multi-layered **artificial neural networks** to automatically learn complex "
                    "patterns in images, audio, and text."
                ),
                "explain_more": (
                    "**Architecture of Deep Networks:**\n"
                    "- **Input Layer**: Takes raw data (pixels, word tokens, audio frequencies).\n"
                    "- **Hidden Layers**: Hierarchically extract features (edges -> textures -> objects).\n"
                    "- **Activation Functions**: Introduce non-linearity (e.g., ReLU, Sigmoid).\n"
                    "- **Backpropagation**: Calculates gradients to update network weights iteratively."
                ),
                "example": (
                    "**Deep Learning Example:** Real-time speech translation apps converting spoken voice frequencies "
                    "directly into translated text in another language."
                ),
                "why": (
                    "**Why Deep Learning:** Traditional ML plateaus when fed massive datasets. Deep learning models "
                    "continue to increase in accuracy as compute and data scale up."
                ),
                "compare": (
                    "**Traditional ML vs. Deep Learning:** Traditional ML requires human experts to engineer features manually. "
                    "Deep Learning extracts features automatically directly from raw inputs."
                ),
                "suggestions": [
                    "Give a Deep Learning example",
                    "What are Neural Networks?",
                    "Explain Overfitting",
                ],
            },
            "overfitting": {
                "default": (
                    "- **Overfitting** happens when a model memorizes training data too closely and performs poorly on new, unseen data.\n"
                    "- **Underfitting** happens when a model is too simple to capture the underlying patterns in the data."
                ),
                "explain_more": (
                    "**How to Prevent Overfitting:**\n"
                    "- **Regularization (L1/L2)**: Penalizing overly large weights.\n"
                    "- **Cross-Validation**: Evaluating performance across multiple data folds.\n"
                    "- **Dropout**: Randomly deactivating neurons during training.\n"
                    "- **Early Stopping**: Halting training before validation loss begins increasing."
                ),
                "example": (
                    "**Analogy for Overfitting:** A student who memorizes practice exam answer keys word-for-word: they score "
                    "100% on the practice test, but fail the real exam because question wording changed."
                ),
                "why": (
                    "**Why Overfitting Matters:** A model that cannot generalize is useless in production. Detecting and "
                    "preventing overfitting is the central challenge in machine learning."
                ),
                "compare": (
                    "**Overfitting (High Variance) vs. Underfitting (High Bias):** Overfitting fits noise and memorizes data. "
                    "Underfitting fails to learn even basic trends because the model is too simple."
                ),
                "suggestions": [
                    "How to prevent overfitting?",
                    "What is Supervised Learning?",
                    "What is Deep Learning?",
                ],
            },
            "supervised_vs_unsupervised": {
                "default": (
                    "**Supervised vs. Unsupervised Learning:**\n"
                    "- **Supervised Learning** trains on *labeled* data (inputs paired with known answers), "
                    "like predicting house prices or classifying spam.\n"
                    "- **Unsupervised Learning** finds hidden patterns in *unlabeled* data, "
                    "like grouping customers into segments (clustering)."
                ),
                "explain_more": (
                    "In **Supervised Learning**, the algorithm is guided by a supervisor (the target labels). "
                    "In **Unsupervised Learning**, the algorithm must find mathematical structure, clusters, or "
                    "density patterns on its own without guidance."
                ),
                "example": (
                    "- **Supervised Example**: Classifying bank transactions as 'Fraud' or 'Legitimate' using past labeled records.\n"
                    "- **Unsupervised Example**: Recommending companion products by discovering items frequently bought together."
                ),
                "why": (
                    "Understanding this difference helps you choose the right algorithm based on whether you possess "
                    "labeled ground truth data or need to explore unstructured datasets."
                ),
                "compare": (
                    "Supervised learning evaluates accuracy against true target labels, while unsupervised learning "
                    "evaluates cluster separation and silhouette metrics."
                ),
                "suggestions": [
                    "Give an example of Supervised Learning",
                    "What is Unsupervised Learning?",
                    "What is Reinforcement Learning?",
                ],
            },
        },
    },
    "programming": {
        "default": (
            "**Programming** is the process of writing clear instructions (code) that a computer "
            "can follow to solve a problem.\n\n"
            "Key building blocks include **variables** (storing data), **conditionals** (making decisions), "
            "**loops** (repeating steps), and **functions** (reusable code blocks)."
        ),
        "explain_more": (
            "**Core Concepts in Software Development:**\n"
            "- **Syntax**: The grammar rules of a programming language.\n"
            "- **Logic**: The step-by-step algorithm designed to solve a task.\n"
            "- **Data Structures**: Formats used to store and organize information efficiently.\n"
            "- **Modularity**: Dividing large software into small, testable, independent components."
        ),
        "example": (
            "**Basic Python Example:**\n"
            "```python\n"
            "# Simple greeting program\n"
            "user_name = input('Enter your name: ')\n"
            "print(f'Hello, {user_name}! Welcome to programming.')\n"
            "```"
        ),
        "why": (
            "**Why Learn Programming:** Programming gives you the power to automate repetitive workflows, build websites "
            "and web apps, analyze data, and create tools that scale to millions of users."
        ),
        "compare": (
            "**Compiled vs. Interpreted Languages:**\n"
            "- **Compiled (C++, Rust)**: Code is converted into machine binaries before execution for high speed.\n"
            "- **Interpreted (Python, JavaScript)**: Code executes line-by-line for fast development and portability."
        ),
        "suggestions": [
            "What is a Python function?",
            "Explain loops in Python",
            "What are variables?",
        ],
        "subtopics": {
            "functions": {
                "default": (
                    "A **function** is a reusable block of code that performs a specific task. "
                    "In Python, functions are defined with `def`:\n\n"
                    "```python\ndef greet(name):\n    return f\"Hello, {name}!\"\n```"
                ),
                "explain_more": (
                    "**Function Components in Python:**\n"
                    "- **Parameters**: Variables defined in the function signature.\n"
                    "- **Arguments**: Actual values passed when calling the function.\n"
                    "- **Return Value**: Output returned to the caller (`return`).\n"
                    "- **Scope**: Variables declared inside a function are local to it."
                ),
                "example": (
                    "**Python Function Example:**\n"
                    "```python\n"
                    "def calculate_total(price, tax_rate=0.08):\n"
                    "    \"\"\"Calculate price after tax.\"\"\"\n"
                    "    return price + (price * tax_rate)\n\n"
                    "final_price = calculate_total(100)\n"
                    "print(f'Total: ${final_price}')  # Total: $108.0\n"
                    "```"
                ),
                "why": (
                    "**Why Use Functions:** Functions eliminate code duplication (DRY principle), improve readability, "
                    "and make applications easy to test and debug."
                ),
                "compare": (
                    "**Function vs. Method:** A function is standalone (`len(data)`), while a method is a function "
                    "attached to an object class (`data.append(x)`)."
                ),
                "suggestions": [
                    "Give a function example",
                    "What are Python loops?",
                    "Why use functions?",
                ],
            },
            "loops": {
                "default": (
                    "A **loop** repeats a block of code multiple times:\n"
                    "- A **`for` loop** iterates over a sequence (like a list or `range`).\n"
                    "- A **`while` loop** repeats as long as a condition stays `True`.\n\n"
                    "Example:\n```python\nfor i in range(3):\n    print(i)\n```"
                ),
                "explain_more": (
                    "**Loop Control Statements:**\n"
                    "- `break`: Immediately exits the loop.\n"
                    "- `continue`: Skips the rest of the current iteration and moves to the next.\n"
                    "- `else`: Executes after loop finishes normally without encountering `break`."
                ),
                "example": (
                    "**Python Loop Examples:**\n"
                    "```python\n"
                    "# For loop with a list\n"
                    "languages = ['Python', 'JavaScript', 'SQL']\n"
                    "for lang in languages:\n"
                    "    print(f'Learning {lang}')\n\n"
                    "# While loop countdown\n"
                    "count = 3\n"
                    "while count > 0:\n"
                    "    print(count)\n"
                    "    count -= 1\n"
                    "```"
                ),
                "why": (
                    "**Why Use Loops:** Loops automate repetitive computing operations without duplicating lines of code."
                ),
                "compare": (
                    "**`for` vs `while` Loops:** Use a `for` loop when iterating over a known sequence or range. "
                    "Use a `while` loop when execution depends on a dynamic condition whose duration is unknown in advance."
                ),
                "suggestions": [
                    "Give a loop example",
                    "What is a function?",
                    "Explain conditionals",
                ],
            },
            "variables": {
                "default": (
                    "A **variable** is a named container that stores a value in memory so you can "
                    "reuse and update it.\n\nExample in Python:\n```python\nname = \"Alice\"\nage = 21\n```"
                ),
                "explain_more": (
                    "**Python Variable Typing:**\n"
                    "Python is **dynamically typed**, meaning you do not need to declare types explicitly. Common types "
                    "include `int`, `float`, `str`, `bool`, `list`, and `dict`."
                ),
                "example": (
                    "**Python Variable Example:**\n"
                    "```python\n"
                    "user_name = 'Sarah'\n"
                    "user_score = 98.5\n"
                    "is_active = True\n\n"
                    "print(f'{user_name} is active: {is_active} with score {user_score}')\n"
                    "```"
                ),
                "why": (
                    "**Why Variables Matter:** Variables allow programs to store dynamic user input, track application state, "
                    "and pass data between different functions."
                ),
                "compare": (
                    "**Variables vs. Constants:** A variable can change its value during execution (`score += 1`). "
                    "A constant represents fixed data that shouldn't change (conventionally named in uppercase: `MAX_RETRIES = 3`)."
                ),
                "suggestions": [
                    "What are Python loops?",
                    "What is a function?",
                    "What are data structures?",
                ],
            },
            "conditionals": {
                "default": (
                    "**Conditionals** (`if`, `elif`, `else`) let your program make decisions "
                    "based on whether a condition is `True` or `False`:\n\n"
                    "```python\nif score >= 50:\n    print(\"Pass\")\nelse:\n    print(\"Try again\")\n```"
                ),
                "explain_more": (
                    "**Boolean Operators in Conditionals:**\n"
                    "- `and`: True only if both conditions are True.\n"
                    "- `or`: True if at least one condition is True.\n"
                    "- `not`: Inverts a boolean condition."
                ),
                "example": (
                    "**Python Conditional Example:**\n"
                    "```python\n"
                    "age = 20\n"
                    "has_ticket = True\n\n"
                    "if age >= 18 and has_ticket:\n"
                    "    print('Access granted.')\n"
                    "elif age >= 18 and not has_ticket:\n"
                    "    print('Please buy a ticket.')\n"
                    "else:\n"
                    "    print('Access denied: Age restricted.')\n"
                    "```"
                ),
                "why": (
                    "**Why Conditionals Matter:** Without conditionals, code could only execute in a single linear sequence. "
                    "Conditionals enable intelligent branching and validation."
                ),
                "compare": (
                    "**`if` vs `elif`:** An `if` statement is always evaluated. An `elif` statement is only evaluated if "
                    "all previous conditions in the block evaluated to False."
                ),
                "suggestions": [
                    "Explain Python loops",
                    "What is a function?",
                    "What are variables?",
                ],
            },
            "data_structures": {
                "default": (
                    "Common **Python data structures** include:\n"
                    "- **List** (`[1, 2, 3]`): ordered, mutable collection.\n"
                    "- **Tuple** (`(1, 2, 3)`): ordered, immutable collection.\n"
                    "- **Dictionary** (`{\"name\": \"Ada\"}`): stores key-value pairs for fast lookup."
                ),
                "explain_more": (
                    "**Choosing Data Structures:**\n"
                    "- Use a **List** when you need an ordered sequence that will change.\n"
                    "- Use a **Tuple** for fixed constants or function return pairs.\n"
                    "- Use a **Dictionary** for fast $O(1)$ key lookups."
                ),
                "example": (
                    "**Python Data Structures Example:**\n"
                    "```python\n"
                    "# Dictionary storing user record\n"
                    "profile = {\n"
                    "    'username': 'coder123',\n"
                    "    'skills': ['Python', 'Flask', 'SQL'],\n"
                    "    'level': 2\n"
                    "}\n"
                    "print(profile['skills'][0])  # Outputs: Python\n"
                    "```"
                ),
                "why": (
                    "**Why Data Structures Matter:** Selecting appropriate data structures dictates both memory consumption "
                    "and the computational speed (time complexity) of your program."
                ),
                "compare": (
                    "**List vs. Dictionary:** Lists are indexed by integers (0, 1, 2). Dictionaries are indexed by keys "
                    "('name', 'id') offering instantaneous direct lookups."
                ),
                "suggestions": [
                    "What is a function?",
                    "Explain OOP in Python",
                    "What are variables?",
                ],
            },
            "oop": {
                "default": (
                    "**Object-Oriented Programming (OOP)** organizes code around **classes** (blueprints) "
                    "and **objects** (instances) that bundle data (attributes) and behavior (methods) together."
                ),
                "explain_more": (
                    "**The Four Pillars of OOP:**\n"
                    "1. **Encapsulation**: Bundling state and methods together.\n"
                    "2. **Abstraction**: Hiding internal mechanics behind clean interfaces.\n"
                    "3. **Inheritance**: Subclasses inheriting properties from parent classes.\n"
                    "4. **Polymorphism**: Different classes implementing the same interface differently."
                ),
                "example": (
                    "**Python OOP Example:**\n"
                    "```python\n"
                    "class Robot:\n"
                    "    def __init__(self, name):\n"
                    "        self.name = name\n\n"
                    "    def speak(self):\n"
                    "        return f'Hello, I am {self.name}!'\n\n"
                    "bot = Robot('Assistant')\n"
                    "print(bot.speak())\n"
                    "```"
                ),
                "why": (
                    "**Why OOP Matters:** OOP structures large applications into manageable, reusable modules that model "
                    "real-world entities."
                ),
                "compare": (
                    "**Class vs. Object:** A Class is the design blueprint (e.g. `Car`). An Object is the actual vehicle "
                    "built from that blueprint (e.g. `my_car = Car()`)."
                ),
                "suggestions": [
                    "Give an OOP example",
                    "What is a function?",
                    "What are data structures?",
                ],
            },
            "debugging": {
                "default": (
                    "**Debugging** is the process of finding and fixing errors (bugs) in code. "
                    "**Syntax errors** occur when code breaks the language's grammar rules, while "
                    "**logic errors** produce unexpected results even when the code runs."
                ),
                "explain_more": (
                    "**Debugging Best Practices:**\n"
                    "- Read the traceback from bottom to top to identify the exact line and error type.\n"
                    "- Use strategic print statements or Python's built-in debugger (`breakpoint()` / `pdb`).\n"
                    "- Write unit tests to verify individual functions against edge cases."
                ),
                "example": (
                    "**Python Exception Handling:**\n"
                    "```python\n"
                    "try:\n"
                    "    val = int('abc')\n"
                    "except ValueError as err:\n"
                    "    print(f'Caught error: {err}')\n"
                    "```"
                ),
                "why": (
                    "**Why Debugging Matters:** Every developer writes bugs; skilled debugging saves countless development "
                    "hours and prevents software crashes."
                ),
                "compare": (
                    "**Syntax vs. Runtime vs. Logic Errors:** Syntax errors prevent execution. Runtime errors crash the program "
                    "mid-flight. Logic errors execute smoothly but compute the wrong answer."
                ),
                "suggestions": [
                    "What is a function?",
                    "What is OOP?",
                    "What are data structures?",
                ],
            },
            "regex": {
                "default": (
                    "A **regular expression (regex)** is a sequence of characters that defines a "
                    "search pattern for matching text—in Python, the built-in `re` module handles regex!"
                ),
                "explain_more": (
                    "**Common Regex Metacharacters:**\n"
                    "- `^` and `$`: Start and end of a line.\n"
                    "- `\\b`: Word boundary.\n"
                    "- `\\d`: Any digit (0-9).\n"
                    "- `.*`: Match any character zero or more times."
                ),
                "example": (
                    "**Python Regex Example:**\n"
                    "```python\n"
                    "import re\n"
                    "text = 'Contact me at test@example.com'\n"
                    "match = re.search(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b', text)\n"
                    "print(match.group())  # test@example.com\n"
                    "```"
                ),
                "why": (
                    "**Why Regex Matters:** Regex provides ultra-fast, robust text parsing, data validation, and intent "
                    "matching without requiring machine learning models."
                ),
                "compare": (
                    "**Regex vs. String Methods:** String methods (`startswith`, `replace`) are simple and fast for exact matches. "
                    "Regex handles complex dynamic patterns, boundaries, and character sets."
                ),
                "suggestions": [
                    "What is a function?",
                    "What are variables?",
                    "What is Python?",
                ],
            },
            "python": {
                "default": (
                    "**Python** is a popular, beginner-friendly programming language known for its "
                    "clean, readable syntax. It is widely used in web development, automation, "
                    "data science, and AI/Machine Learning."
                ),
                "explain_more": (
                    "Python emphasizes code readability, provides an extensive standard library (batteries included), "
                    "and has the world's largest ecosystem for artificial intelligence and data analytics (NumPy, PyTorch, Pandas)."
                ),
                "example": (
                    "**Python Example:**\n"
                    "```python\n"
                    "languages = ['Python', 'SQL', 'HTML']\n"
                    "print([lang.upper() for lang in languages])\n"
                    "# Outputs: ['PYTHON', 'SQL', 'HTML']\n"
                    "```"
                ),
                "why": (
                    "**Why Python:** Its syntax reads closely to English, making it rapid to learn, fast to prototype in, "
                    "and backed by millions of open-source packages."
                ),
                "compare": (
                    "**Python vs. JavaScript:** Python is widely preferred for backend services, data analysis, and AI. "
                    "JavaScript is the universal language of web browsers and client-side interfaces."
                ),
                "suggestions": [
                    "What is a Python function?",
                    "Explain loops in Python",
                    "What are variables?",
                ],
            },
        },
    },
}

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
        "Python and Flask for an internship project. "
        "I recognize your questions using Python functions and regular expressions—"
        "no paid APIs required!"
    ),
    "capabilities": (
        "Here is what I can help you with:\n"
        "- **Greetings & Casual Chat** (saying hello, thanks, or goodbye)\n"
        "- **Artificial Intelligence** (AI basics, NLP, computer vision, generative AI)\n"
        "- **Machine Learning** (ML basics, supervised/unsupervised/reinforcement learning, neural networks)\n"
        "- **Basic Programming** (Python, variables, loops, functions, conditionals, data structures, OOP, debugging)\n"
        "- **Contextual Follow-ups** (Explain more, Give an example, Why?, Compare them)\n"
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
        "You can also ask follow-up questions like **'Explain more'**, **'Give an example'**, "
        "**'Why?'**, or **'Compare them'**!"
    ),
    "unknown": (
        "I'm not sure how to answer that yet—I'm a rule-based chatbot focused on "
        "**Artificial Intelligence**, **Machine Learning**, and **basic programming**.\n\n"
        "Try asking something like *\"What is AI?\"*, *\"Explain machine learning\"*, "
        "or *\"What is a loop in Python?\"*, or type **help** for more ideas."
    ),
}

DEFAULT_SUGGESTIONS: dict[str, list[str]] = {
    "greeting": [
        "What is Artificial Intelligence?",
        "What is Machine Learning?",
        "What can you do?",
    ],
    "identity": [
        "What can you do?",
        "What is Artificial Intelligence?",
        "Help",
    ],
    "capabilities": [
        "What is AI?",
        "What is Machine Learning?",
        "What is a Python function?",
    ],
    "thanks": [
        "What is Supervised Learning?",
        "What is a Python loop?",
        "What is AI?",
    ],
    "goodbye": [
        "Hello again!",
        "What is Machine Learning?",
        "What is a Python function?",
    ],
    "help": [
        "What is AI?",
        "Explain supervised learning",
        "What is a Python function?",
    ],
    "unknown": [
        "What is Artificial Intelligence?",
        "What is Machine Learning?",
        "What is a Python function?",
    ],
    "empty": [
        "What is Artificial Intelligence?",
        "What is Machine Learning?",
        "What can you do?",
    ],
}

# ============================================================================
# 3. Context Management (In-Memory Per-Conversation Store)
# ============================================================================

# Tracks context separately for each conversation_id to avoid cross-chat leakage
CONVERSATION_CONTEXTS: dict[str, dict[str, Any]] = {}


def get_conversation_context(conversation_id: str | None) -> dict[str, Any]:
    """Retrieve the context dictionary for a given conversation_id."""
    if not conversation_id or conversation_id not in CONVERSATION_CONTEXTS:
        return {"topic": None, "subtopic": None, "intent": None}
    return dict(CONVERSATION_CONTEXTS[conversation_id])


def update_conversation_context(
    conversation_id: str | None, context_update: dict[str, Any]
) -> dict[str, Any]:
    """Update and persist context for a conversation_id."""
    if not conversation_id:
        return context_update
    ctx = CONVERSATION_CONTEXTS.setdefault(
        conversation_id, {"topic": None, "subtopic": None, "intent": None}
    )
    ctx.update(context_update)
    return dict(ctx)


def clear_conversation_context(conversation_id: str | None) -> None:
    """Clear memory context for a deleted or cleared conversation."""
    if conversation_id and conversation_id in CONVERSATION_CONTEXTS:
        del CONVERSATION_CONTEXTS[conversation_id]


# ============================================================================
# 4. Engine Processing Logic
# ============================================================================


def clean_input(user_message: str | None) -> str:
    """Safely normalize user input by trimming and collapsing whitespace."""
    if not isinstance(user_message, str):
        return ""
    return " ".join(user_message.strip().split())


def detect_followup(cleaned_text: str) -> str | None:
    """Detect if the message is a contextual follow-up like 'Explain more' or 'Give an example'."""
    for followup_name, pattern in FOLLOWUP_PATTERNS:
        if pattern.search(cleaned_text):
            return followup_name
    return None


def detect_subtopic(cleaned_text: str, topic: str) -> str | None:
    """Identify matching subtopic within a known topic category."""
    subtopics = SUBTOPIC_PATTERNS.get(topic, [])
    for subtopic_key, pattern in subtopics:
        if pattern.search(cleaned_text):
            return subtopic_key
    return None


def detect_intent(user_message: str | None) -> str:
    """Identify the primary intent for a user message."""
    cleaned_text = clean_input(user_message)
    if not cleaned_text:
        return "empty"

    for intent_name, pattern in INTENT_PATTERNS:
        if pattern.search(cleaned_text):
            return intent_name

    # If it is a standalone follow-up, categorize as followup
    followup = detect_followup(cleaned_text)
    if followup:
        return f"followup_{followup}"

    return "unknown"


def _handle_context_followup(
    followup_type: str,
    context: dict[str, Any],
) -> tuple[str, str, list[str], dict[str, Any]]:
    """Generate response for a follow-up using previous conversation context."""
    topic = context.get("topic")
    subtopic = context.get("subtopic")

    # If there is no prior topic in context
    if not topic or topic not in KNOWLEDGE_BASE:
        msg = (
            "I don't have enough context yet. Please ask a question about "
            "**Artificial Intelligence**, **Machine Learning**, or **Python programming** first, "
            "and then I can provide examples, comparisons, or deeper explanations!"
        )
        suggestions = [
            "What is Artificial Intelligence?",
            "What is Machine Learning?",
            "What is a Python function?",
        ]
        return msg, f"followup_{followup_type}", suggestions, context

    topic_data = KNOWLEDGE_BASE[topic]
    subtopic_data = None
    if subtopic and "subtopics" in topic_data and subtopic in topic_data["subtopics"]:
        subtopic_data = topic_data["subtopics"][subtopic]

    target_data = subtopic_data if subtopic_data is not None else topic_data

    # Retrieve response according to follow-up kind
    if followup_type in target_data:
        response_text = target_data[followup_type]
    else:
        response_text = target_data.get("default", topic_data.get("default", ""))

    suggestions = target_data.get("suggestions", topic_data.get("suggestions", []))

    # Maintain existing context
    new_context = {
        "topic": topic,
        "subtopic": subtopic,
        "intent": f"followup_{followup_type}",
    }

    return response_text, f"followup_{followup_type}", suggestions, new_context


def get_bot_response_details(
    user_message: str | None,
    context: dict[str, Any] | None = None,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """Primary engine function returning full details: response, intent, suggestions, context."""
    cleaned_text = clean_input(user_message)
    current_context = (
        dict(context)
        if isinstance(context, dict)
        else get_conversation_context(conversation_id)
    )

    if not cleaned_text:
        return {
            "response": STATIC_RESPONSES["empty"],
            "intent": "empty",
            "suggestions": DEFAULT_SUGGESTIONS["empty"],
            "context": current_context,
        }

    # 1. Check if user input is an explicit topic intent (takes precedence over pure follow-up)
    intent = "unknown"
    for intent_name, pattern in INTENT_PATTERNS:
        if pattern.search(cleaned_text):
            intent = intent_name
            break

    # 2. If it's a known knowledge-base topic
    if intent in KNOWLEDGE_BASE:
        topic = intent
        subtopic = detect_subtopic(cleaned_text, topic)
        topic_data = KNOWLEDGE_BASE[topic]

        # Check if the user specifically asked for an example inside the topic query
        # e.g., "Give an example of a Python function"
        is_asking_example = bool(re.search(r"\b(?:example|sample|code)\b", cleaned_text, re.IGNORECASE))
        is_asking_comparison = bool(re.search(r"\b(?:compare|difference|vs)\b", cleaned_text, re.IGNORECASE))

        if subtopic and "subtopics" in topic_data and subtopic in topic_data["subtopics"]:
            sub_data = topic_data["subtopics"][subtopic]
            if is_asking_example and "example" in sub_data:
                response_text = sub_data["example"]
            elif is_asking_comparison and "compare" in sub_data:
                response_text = sub_data["compare"]
            else:
                response_text = sub_data.get("default", topic_data["default"])
            suggestions = sub_data.get("suggestions", topic_data["suggestions"])
        else:
            if is_asking_example and "example" in topic_data:
                response_text = topic_data["example"]
            elif is_asking_comparison and "compare" in topic_data:
                response_text = topic_data["compare"]
            else:
                response_text = topic_data["default"]
            suggestions = topic_data.get("suggestions", [])

        new_context = {
            "topic": topic,
            "subtopic": subtopic,
            "intent": intent,
        }
        if conversation_id:
            update_conversation_context(conversation_id, new_context)

        return {
            "response": response_text,
            "intent": intent,
            "suggestions": suggestions,
            "context": new_context,
        }

    # 3. Check if it's a contextual follow-up (e.g. "Explain more", "Give an example", "Why?", "Compare them")
    followup_type = detect_followup(cleaned_text)
    if followup_type:
        response_text, followup_intent, suggestions, new_context = (
            _handle_context_followup(followup_type, current_context)
        )
        if conversation_id:
            update_conversation_context(conversation_id, new_context)
        return {
            "response": response_text,
            "intent": followup_intent,
            "suggestions": suggestions,
            "context": new_context,
        }

    # 4. Handle conversational and static intents
    if intent in STATIC_RESPONSES:
        response_text = STATIC_RESPONSES[intent]
        suggestions = DEFAULT_SUGGESTIONS.get(intent, DEFAULT_SUGGESTIONS["unknown"])
        new_context = dict(current_context)
        new_context["intent"] = intent
        if conversation_id:
            update_conversation_context(conversation_id, new_context)
        return {
            "response": response_text,
            "intent": intent,
            "suggestions": suggestions,
            "context": new_context,
        }

    # 5. Unknown query fallback
    new_context = dict(current_context)
    new_context["intent"] = "unknown"
    if conversation_id:
        update_conversation_context(conversation_id, new_context)

    return {
        "response": STATIC_RESPONSES["unknown"],
        "intent": "unknown",
        "suggestions": DEFAULT_SUGGESTIONS["unknown"],
        "context": new_context,
    }


def get_bot_response(
    user_message: str | None,
    context: dict[str, Any] | None = None,
    conversation_id: str | None = None,
) -> str:
    """Backward-compatible response helper returning string response."""
    details = get_bot_response_details(user_message, context=context, conversation_id=conversation_id)
    return details["response"]
