# Student Support AI

COMP 472 Mini Project 1: Intelligent Support Assistant with Sentiment Analysis and Retrieval.

## Files

- `support_assistant.py`: main Python chatbot program
- `knowledge_base.csv`: questions and answers used by the assistant
- `requirements.txt`: required Python packages
- `reflection_template.txt`: short reflection template
- `sample_run.txt`: example demo output
- `program_execution_screenshot.png`: sample screenshot-style image of execution

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
# venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## Run

```bash
python support_assistant.py
```

You can also provide a CSV path manually:

```bash
python support_assistant.py knowledge_base.csv
```

Type `quit` to exit.

## Notes for Demo

Be ready to explain:

1. How `pandas` loads the CSV file.
2. How `SentenceTransformer` converts questions into embeddings.
3. How cosine similarity finds the closest knowledge-base question.
4. How the `transformers` sentiment-analysis pipeline detects sentiment.
5. Why negative sentiment with confidence above 0.90 triggers escalation.
