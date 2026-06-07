import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


knowledge_base_path = "knowledge_base.csv"

# Small, fast sentence-transformer model for generating embeddings. This
# default works well for demo / small-scale semantic search tasks.
embedding_model_name = "all-MiniLM-L6-v2"


# Runtime state -----------------------------------------------------
# Stored lists of knowledge-base entries. Populated by load_knowledge_base().
questions = []
answers = []

# Numpy array of embeddings corresponding to `questions`. Populated by
# generate_embeddings(). Kept as a global for simplicity.
question_embeddings = None

# Simple in-memory conversation log: list of tuples (user_query, sentiment_label,
# answer_returned, sentiment_confidence). This is kept in RAM only and not
# persisted anywhere by this script.
conversation_history = []


# Model handles (initialized in setup_assistant) ---------------------
embedding_model = None
sentiment_analyzer = None


def load_knowledge_base():
    """Load questions and answers from `knowledge_base_path` CSV.

    Expects a CSV with at least two columns named 'question' and 'answer'.

    Side-effects:
    - Sets the module-level `questions` and `answers` lists.

    Raises:
    - FileNotFoundError: if the CSV path doesn't exist.
    - ValueError: if required columns are missing or there's no data.

    Why explicit validation: downstream code assumes a non-empty list of
    strings (so we coerce to str and drop NaNs up-front).
    """

    global questions, answers

    path = Path(knowledge_base_path)

    # Fail early with a clear message if the file is missing.
    if not path.exists():
        raise FileNotFoundError(
            f"Knowledge base file not found: {path}. "
            "Make sure knowledge_base.csv is in the same folder as this script."
        )

    # Read the CSV using pandas. This lets users supply larger KBs easily.
    data = pd.read_csv(path)

    # Ensure the CSV has the expected columns. This is a common user error.
    if "question" not in data.columns or "answer" not in data.columns:
        raise ValueError("CSV file must contain 'question' and 'answer' columns.")

    # Remove rows where either question or answer is missing.
    data = data.dropna(subset=["question", "answer"])

    # If the file contains no usable rows, raise so caller can handle it.
    if data.empty:
        raise ValueError("Knowledge base is empty. Add at least one question and answer.")

    # Normalize to Python strings and save into module-global lists.
    questions = data["question"].astype(str).tolist()
    answers = data["answer"].astype(str).tolist()


def generate_embeddings():
    """Generate embeddings for all knowledge-base questions.

    Side-effects:
    - Sets the module-level `question_embeddings` (typically a numpy array).

    Notes:
    - This function relies on `embedding_model` being initialized (see
      `setup_assistant`). If it's None, this will raise an AttributeError.
    - We keep all embeddings in memory for fast similarity computation. For
      large KBs consider batching, on-disk stores, or approximate nearest
      neighbors (e.g., FAISS).
    """

    global question_embeddings

    # The model's `encode` returns a ndarray of shape (n_questions, dim).
    question_embeddings = embedding_model.encode(questions)


def retrieve_answer(user_query):
    """Find the best-matching answer from the knowledge base for a query.

    Args:
        user_query (str): The raw user text to match.

    Returns:
        tuple(answer:str, similarity:float, matched_question:str)

    Raises:
        RuntimeError: if embeddings haven't been generated yet.

    Implementation details:
    - Encode the single query with the same embedding model used for the KB.
    - Compute cosine similarity between the query embedding and all stored
      question embeddings.
    - Choose the highest similarity as the best match.

    Edge cases:
    - If similarity is low, the returned answer may not actually address the
      user's intent; this script prints the similarity so callers can decide
      to re-ask or fallback to a human.
    """

    if question_embeddings is None:
        # The caller is expected to call generate_embeddings() during setup.
        raise RuntimeError("Question embeddings were not generated.")

    # Encode the user's query into the same embedding space. encode(...) can
    # accept a list of strings; we pass a single-element list and then use the
    # first row of the result.
    query_embedding = embedding_model.encode([user_query])

    # cosine_similarity accepts 2D arrays and returns a matrix. We want the
    # similarity between the single query and every KB question, so we take
    # the first row of the result.
    similarities = cosine_similarity(query_embedding, question_embeddings)[0]

    # argmax returns the index of the highest scoring KB question. We coerce
    # to int for consistency and to avoid numpy types leaking out.
    best_index = int(np.argmax(similarities))

    # Gather the selected answer and metadata to return to the caller.
    answer = answers[best_index]
    similarity = float(similarities[best_index])
    matched_question = questions[best_index]

    return answer, similarity, matched_question


def analyze_sentiment(text):
    """Run a sentiment analysis model on the provided text.

    Returns a tuple (label, confidence) where label is typically 'POSITIVE' or
    'NEGATIVE' depending on the underlying transformers pipeline model, and
    confidence is a float score between 0 and 1.

    The function expects `sentiment_analyzer` to be a `transformers` pipeline
    initialized for sentiment analysis (see setup_assistant).
    """

    result = sentiment_analyzer(text)[0]

    # The pipeline returns objects like {'label': 'POSITIVE', 'score': 0.99}
    label = str(result["label"])
    confidence = float(result["score"])

    return label, confidence


def should_escalate(label, confidence):
    """Simple heuristic: escalate when sentiment is NEGATIVE with high confidence.

    This threshold is intentionally conservative (0.90) to avoid false
    positives. Teams can tune the label and threshold depending on their
    needs (e.g., add anger detection, profanity checks, or domain-specific
    classifiers).
    """

    return label.upper() == "NEGATIVE" and confidence > 0.90


def answer_question(user_query):
    """High-level handler for processing a single user query.

    Steps:
    - Analyze sentiment of the raw user input.
    - Retrieve the best KB answer via semantic search.
    - Print the sentiment, recommended escalation (if any), the answer, the
      matched KB question, and the similarity score.
    - Append a record to `conversation_history` for potential later use.

    This function performs printing directly (suitable for the provided REPL
    `chat()`), but it could be refactored to return a serializable object
    for use in other contexts (e.g., a web API) without printing.
    """

    label, confidence = analyze_sentiment(user_query)
    answer, similarity, matched_question = retrieve_answer(user_query)

    # Present sentiment info first so operators can notice worrying language.
    print(f"Sentiment: {label} ({confidence:.2f})")

    if should_escalate(label, confidence):
        # A human-in-the-loop recommendation for strongly negative inputs.
        print("Recommended escalation: Contact human advisor.")

    # Present the selected answer and helpful metadata for debugging.
    print(f"Answer: {answer}")
    print(f"Matched knowledge-base question: {matched_question}")
    print(f"Semantic similarity: {similarity:.2f}")

    # Store the exchange in memory. Useful for offline analysis or replay.
    conversation_history.append((user_query, label, answer, confidence))


def chat():
    """A simple REPL to interact with the assistant via the terminal.

    The loop reads user input, allows 'quit' to exit, and delegates handling
    to answer_question(). It also gracefully handles KeyboardInterrupt to let
    users Ctrl-C out of the loop.
    """

    print("Welcome to Student Support AI")
    print("Type 'quit' to exit.")

    while True:
        try:
            user_query = input("You: ").strip()

            if user_query.lower() == "quit":
                print("Goodbye!")
                break

            if not user_query:
                # Empty input: prompt user again without performing analysis.
                print("Please enter a question or type 'quit' to exit.")
                continue

            answer_question(user_query)

        except KeyboardInterrupt:
            # Allow the user to exit cleanly with Ctrl-C.
            print("\nGoodbye!")
            break

        except Exception as error:
            # Generic error handling so the REPL stays live. For debugging,
            # catch and display the exception to the terminal.
            print(f"Error: {error}")


def setup_assistant():
    """Prepare models and embeddings before starting interaction.

    Actions performed:
    - Loads the knowledge base CSV into memory.
    - Initializes the SentenceTransformer used for embeddings.
    - Initializes a transformers sentiment-analysis pipeline.
    - Generates embeddings for all KB questions.

    This is separated from `main()` to keep initialization logic testable.
    """

    global embedding_model, sentiment_analyzer

    load_knowledge_base()

    # Load the embedding model. On first run this will download model weights
    # (if not already cached) which may take several seconds.
    embedding_model = SentenceTransformer(embedding_model_name)

    # Initialize a default sentiment-analysis pipeline. The model used is the
    # default for the installed transformers version; teams can pass a model
    # name to pipeline(...) if they need a different classifier.
    sentiment_analyzer = pipeline("sentiment-analysis")

    # Pre-compute embeddings so retrieval is fast in the chat loop.
    generate_embeddings()


def main():
    """Entry point when running the script as a program.

    Usage:
        python support_assistant.py [path/to/knowledge_base.csv]

    If a command-line argument is provided it is used as the KB path; this
    allows running against a different CSV without editing the file.
    """

    global knowledge_base_path

    if len(sys.argv) > 1:
        knowledge_base_path = sys.argv[1]

    setup_assistant()
    chat()


if __name__ == "__main__":
    main()