import nltk
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample training data
questions = [
    "hi", "hello", "hey",
    "what services do you offer",
    "how can you help me",
    "bye", "goodbye"
]

answers = [
    "Hello! How can I help you?",
    "Hello! How can I help you?",
    "Hello! How can I help you?",
    "We provide customer support and technical assistance.",
    "I can answer your queries and provide support.",
    "Goodbye! Have a nice day.",
    "Goodbye! Have a nice day."
]

# Convert text to vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(questions)

# Train model
model = MultinomialNB()
model.fit(X, answers)

print("AI Chatbot is ready! Type 'exit' to stop.")

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break

    user_vec = vectorizer.transform([user_input])
    response = model.predict(user_vec)
    print("Bot:", response[0])

