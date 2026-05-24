import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


@dataclass
class SupportAssistant:
    """AI-powered student support assistant.

    This class wraps a small retrieval+sentiment assistant using a sentence embedding
    model (from sentence-transformers) for semantic search and a transformers
    sentiment-analysis pipeline for simple sentiment detection.

    Attributes:
        knowledge_base_path: Path to a CSV file with 'question' and 'answer' columns.
        embedding_model_name: Name of the sentence-transformers model used to embed text.
        questions: Loaded questions from the knowledge base (keeps order with answers).
        answers: Loaded answers from the knowledge base.
        question_embeddings: NumPy array of embeddings (shape: num_questions x embedding_dim).
        conversation_history: List of tuples (user_query, sentiment_label, answer, confidence).
    """

    # Default configuration values are intentionally simple — easy to override in tests
    knowledge_base_path: str = "knowledge_base.csv"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    questions: List[str] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)
    question_embeddings: np.ndarray | None = None
    conversation_history: List[Tuple[str, str, str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Post-initialization: load KB, instantiate models, and precompute embeddings.

        Rationale: keep initialization simple so a caller who constructs the class
        immediately has a ready-to-use assistant. This does mean heavy work (model
        downloads/initialization) happens at construction time; in larger apps you
        might prefer lazy-loading or dependency injection.
        """
        # Load CSV knowledge base into memory (questions/answers lists)
        self.load_knowledge_base()

        # Initialize the embedding model once and reuse it for all queries to save time
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Use transformers pipeline for sentiment analysis (convenient default). In
        # production you should pin a model and set HF token for rate limits.
        self.sentiment_analyzer = pipeline("sentiment-analysis")

        # Precompute embeddings for the knowledge-base questions to make retrieval cheap
        self.generate_embeddings()

    def load_knowledge_base(self) -> None:
        """Load student questions and answers from a CSV file.

        The CSV must contain 'question' and 'answer' columns. Rows with missing
        values in these columns are dropped. The function sets `self.questions`
        and `self.answers` as parallel lists in the same order they appear in the file.

        Raises:
            FileNotFoundError: if the file does not exist.
            ValueError: if required columns are missing or the KB is empty after cleaning.
        """
        path = Path(self.knowledge_base_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Knowledge base file not found: {path}. "
                "Make sure knowledge_base.csv is in the same folder as this script."
            )

        # Use pandas for robust CSV parsing (handles quoted fields, line endings, etc.)
        data = pd.read_csv(path)
        required_columns = {"question", "answer"}
        if not required_columns.issubset(data.columns):
            raise ValueError("CSV file must contain 'question' and 'answer' columns.")

        # Drop rows where either question or answer is missing — those are not useful
        data = data.dropna(subset=["question", "answer"])
        if data.empty:
            raise ValueError("Knowledge base is empty. Add at least one question and answer.")

        # Convert to plain Python lists of strings for downstream processing
        self.questions = data["question"].astype(str).tolist()
        self.answers = data["answer"].astype(str).tolist()

    def generate_embeddings(self) -> None:
        """Convert stored questions into sentence embeddings.

        The model encodes the list of questions into a 2D NumPy array. We store the
        embeddings so retrieval queries can compute similarity quickly without
        re-encoding the KB each time.
        """
        # The SentenceTransformer.encode method returns a NumPy array by default
        self.question_embeddings = self.embedding_model.encode(self.questions)

    def retrieve_answer(self, user_query: str) -> Tuple[str, float, str]:
        """Return the best-matching answer for a user query.

        Args:
            user_query: The user's free-text question.

        Returns:
            A tuple of (answer, similarity_score, matched_question) where
            similarity_score is the cosine similarity between query and matched question.

        Raises:
            RuntimeError: if question embeddings have not been generated.
        """
        if self.question_embeddings is None:
            raise RuntimeError("Question embeddings were not generated.")

        # Encode the single query; result is shape (1, dim)
        query_embedding = self.embedding_model.encode([user_query])

        # Cosine similarity between the query and all precomputed KB question embeddings
        similarities = cosine_similarity(query_embedding, self.question_embeddings)[0]

        # Choose the index with highest similarity
        best_index = int(np.argmax(similarities))

        return self.answers[best_index], float(similarities[best_index]), self.questions[best_index]

    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """Detect sentiment label and confidence score for a piece of text.

        Args:
            text: The input text to analyze.

        Returns:
            A tuple (label, confidence) where label is the detected sentiment label
            (e.g., 'POSITIVE' or 'NEGATIVE') and confidence is a float score in [0,1].
        """
        # The transformers pipeline returns a list of results even for a single input
        result = self.sentiment_analyzer(text)[0]
        label = str(result["label"])
        confidence = float(result["score"])
        return label, confidence

    @staticmethod
    def should_escalate(label: str, confidence: float) -> bool:
        """Decide whether to recommend escalation to a human.

        Heuristic: escalate when sentiment is strongly negative (label is 'NEGATIVE'
        and confidence is > 0.90). This is intentionally conservative to avoid
        false positives.
        """
        return label.upper() == "NEGATIVE" and confidence > 0.90

    def answer_question(self, user_query: str) -> None:
        """Full pipeline for answering a single user query.

        Steps:
        1. Analyze sentiment of the query.
        2. Retrieve the most semantically similar KB answer.
        3. Print the results and append the interaction to conversation history.

        Args:
            user_query: Text of the user's question.
        """
        label, confidence = self.analyze_sentiment(user_query)
        answer, similarity, matched_question = self.retrieve_answer(user_query)

        # Output: simple textual feedback for a CLI assistant
        print(f"Sentiment: {label} ({confidence:.2f})")
        if self.should_escalate(label, confidence):
            print("Recommended escalation: Contact human advisor.")
        print(f"Answer: {answer}")
        print(f"Matched knowledge-base question: {matched_question}")
        print(f"Semantic similarity: {similarity:.2f}")

        # Track the interaction for possible auditing or later use
        self.conversation_history.append((user_query, label, answer, confidence))

    def chat(self) -> None:
        """Run the interactive conversation loop (CLI).

        The loop reads user lines from stdin. Typing 'quit' (case-insensitive)
        exits the loop. We handle KeyboardInterrupt to allow Ctrl-C graceful exit.
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
                    print("Please enter a question or type 'quit' to exit.")
                    continue
                self.answer_question(user_query)
            except KeyboardInterrupt:
                # Allow Ctrl-C to exit the chat loop cleanly
                print("\nGoodbye!")
                break
            except Exception as error:
                # Catch-all to avoid crashing the REPL; in production log this instead
                print(f"Error: {error}")


def main() -> None:
    """Create and run the assistant."""
    kb_path = sys.argv[1] if len(sys.argv) > 1 else "knowledge_base.csv"
    assistant = SupportAssistant(knowledge_base_path=kb_path)
    assistant.chat()


if __name__ == "__main__":
    main()
