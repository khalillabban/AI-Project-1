import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class SpamDetectorAI:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = None
        self.vectorizer = TfidfVectorizer()
        #Using Logistic Regression
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced") #since so many more ham than spam messages, we need to balance the classes to prevent bias toward ham

    def load_dataset(self):
        try:
            self.data = pd.read_csv(self.dataset_path, encoding="latin-1")

            self.data = self.data.iloc[:, 0:2] #removing extra columns
            self.data.columns = ["label", "message"]
            self.data = self.data.dropna() #dropping null entries, if any

            print("Dataset loaded successfully.")

        except FileNotFoundError:
            print("Error: Dataset file not found.")
            exit()

        except Exception as e:
            print("Error loading dataset:", e)
            exit()

    #creating the visualization of the dataset, showing the distribution of spam vs ham messages
    def show_data_chart(self):
        counts = self.data["label"].value_counts()

        counts.plot(kind="bar")

        plt.title("Spam vs Ham Messages")
        plt.xlabel("Message Type")
        plt.ylabel("Number of Messages")
        plt.xticks(rotation=0)
        plt.tight_layout()

        plt.savefig("spam_chart.png")
        plt.close()

        print("\nChart saved as spam_chart.png")

    def train_model(self):
        X = self.data["message"]
        y = self.data["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        X_train_numbers = self.vectorizer.fit_transform(X_train)
        X_test_numbers = self.vectorizer.transform(X_test)

        self.model.fit(X_train_numbers, y_train)

        predictions = self.model.predict(X_test_numbers)

        accuracy = accuracy_score(y_test, predictions)
        matrix = confusion_matrix(y_test, predictions)

        print("\nModel training complete.")
        print(f"Accuracy: {accuracy * 100:.2f}%")

        print("\nConfusion Matrix:")
        print(matrix)

        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

    def predict_message(self, message):
        converted_message = self.vectorizer.transform([message])

        prediction = self.model.predict(converted_message)[0]

        probabilities = self.model.predict_proba(converted_message)[0]
        confidence = max(probabilities) * 100

        print(f"Prediction: {prediction.upper()}")
        print(f"Confidence: {confidence:.2f}%")

    def interactive_loop(self):
        print("\nType 'quit' to exit.")

        while True:
            message = input("\nEnter message: ")

            if message.lower() == "quit":
                print("Goodbye!")
                break

            if message.strip() == "":
                print("Please enter a message.")
                continue

            self.predict_message(message)


def main():
    print("Welcome to Spam Detection AI")
    print("Training model...\n")

    detector = SpamDetectorAI("spam.csv")

    detector.load_dataset()
    detector.show_data_chart()
    detector.train_model()
    detector.interactive_loop()


if __name__ == "__main__":
    main()