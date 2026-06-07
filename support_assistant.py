import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


knowledge_base_path = "knowledge_base.csv"

#Small and fast sentencer transformer model for generating embeddings
embedding_model_name = "all-MiniLM-L6-v2"


#Populated from knowledge_base.csv
questions = []
answers = []

question_embeddings = None

conversation_history = []

#Initialized in setup_assistant()
embedding_model = None
sentiment_analyzer = None


def load_knowledge_base():
    global questions, answers

    path = Path(knowledge_base_path)

    # Fail early with a clear message if the file is missing.
    if not path.exists():
        raise FileNotFoundError(
            f"Knowledge base file not found: {path}. "
            "Make sure knowledge_base.csv is in the same folder as this script."
        )

    # Loading the data into a dataframe using pandas
    data = pd.read_csv(path)

    # Ensuring the CSV has the expected columns
    if "question" not in data.columns or "answer" not in data.columns:
        raise ValueError("CSV file must contain 'question' and 'answer' columns.")

    # Remove rows where either question or answer is missing.
    data = data.dropna(subset=["question", "answer"])

    # If the file contains no usable rows, raise error so the caller can handle it.
    if data.empty:
        raise ValueError("Knowledge base is empty. Add at least one question and answer.")

    # Normalize to Python strings and save into module-global lists.
    questions = data["question"].astype(str).tolist()
    answers = data["answer"].astype(str).tolist()


def generate_embeddings():
    global question_embeddings

    # The model's `encode` returns a ndarray of shape (n_questions, dim).
    question_embeddings = embedding_model.encode(questions) 


def retrieve_answer(user_query):
    if question_embeddings is None:
        # The caller is expected to call generate_embeddings() during setup.
        raise RuntimeError("Question embeddings were not generated.")

    #Encodes the user's query
    query_embedding = embedding_model.encode([user_query])

    #cosine_similarity computes the cosine similarity between the query embedding and each of the question embeddings, resulting in a list of similarity scores.
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
    result = sentiment_analyzer(text)[0]

    raw_label = str(result["label"]).upper()
    confidence = float(result["score"])

    if raw_label in ["LABEL_0", "NEGATIVE"]:
        label = "NEGATIVE"
    elif raw_label in ["LABEL_1", "NEUTRAL"]:
        label = "NEUTRAL"
    elif raw_label in ["LABEL_2", "POSITIVE"]:
        label = "POSITIVE"
    else:
        label = raw_label

    return label, confidence


def should_escalate(label, confidence):
    # Heuristic: escalate if negative sentiment is detected with high confidence.
    return label.upper() == "NEGATIVE" and confidence > 0.90


def answer_question(user_query):
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
    global embedding_model, sentiment_analyzer

    load_knowledge_base()

    #Loads the embedding model
    embedding_model = SentenceTransformer(embedding_model_name)

    sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

    # Pre-compute embeddings so retrieval is fast in the chat loop.
    generate_embeddings()


def main():
    global knowledge_base_path

    if len(sys.argv) > 1:
        knowledge_base_path = sys.argv[1]

    setup_assistant()
    chat()


if __name__ == "__main__":
    main()