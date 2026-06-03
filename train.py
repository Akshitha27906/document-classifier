from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
dataset = load_dataset("fancyzhx/ag_news")

# Preprocess
train_texts = [t.lower().strip() for t in dataset["train"]["text"]]
test_texts = [t.lower().strip() for t in dataset["test"]["text"]]
train_labels = list(dataset["train"]["label"])
test_labels = list(dataset["test"]["label"])

# Convert words to numbers
print("Converting text to numbers...")
vectorizer = TfidfVectorizer(max_features=50000)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Train the model
print("Training the model... (takes 1-2 minutes)")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, train_labels)
print("Training complete!")

# Test accuracy
print("Testing accuracy...")
predictions = model.predict(X_test)
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Predict new articles
def predict(text):
    label_names = ["World", "Sports", "Business", "Science/Tech"]
    cleaned = text.lower().strip()
    vector = vectorizer.transform([cleaned])
    result = model.predict(vector)[0]
    print(f"Text: {text}")
    print(f"Category: {label_names[result]}")
    print()

predict("NASA launches new rocket to the moon")
predict("Manchester United wins the championship")
predict("Apple releases new iPhone model")
predict("World leaders meet for climate summit")
# Let user type anything!
print("=" * 50)
print("  TRY IT YOURSELF!")
print("=" * 50)

while True:
    text = input("\nType a sentence (or 'quit' to exit): ")
    if text.lower() == "quit":
        print("Goodbye!")
        break
    predict(text)
