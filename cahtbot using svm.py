import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Define the data for hypertension and diabetes
hypertension_data = [
    "What are the symptoms of hypertension?",
    "What are the causes of hypertension?",
    "What are the treatments for hypertension?",
    "What are the complications of hypertension?"
]

diabetes_data = [
    "What are the symptoms of diabetes?",
    "What are the causes of diabetes?",
    "What are the treatments for diabetes?",
    "What are the complications of diabetes?"
]

# Preprocess the data
def preprocess(data):
    processed_data = []
    for sentence in data:
        words = word_tokenize(sentence.lower())
        processed_data.append(' '.join(words))
    return processed_data

hypertension_data = preprocess(hypertension_data)
diabetes_data = preprocess(diabetes_data)

# Combine the data and create labels
all_data = hypertension_data + diabetes_data
labels = ['hypertension']*len(hypertension_data) + ['diabetes']*len(diabetes_data)

# Convert the data to feature vectors using TfidfVectorizer
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(all_data)

# Train the SVM model
model = SVC(kernel='linear')
model.fit(features, labels)

# Define the chatbot function
def chatbot(text):
    # Preprocess the user's question
    processed_text = preprocess([text])
    # Convert the question to a feature vector
    features_text = vectorizer.transform(processed_text)
    # Use the SVM model to predict the label of the question
    label = model.predict(features_text)[0]
    # Return a response based on the label
    if label == 'hypertension':
        return "Here's some information on hypertension."
    elif label == 'diabetes':
        return "Here's some information on diabetes."
    else:
        return "Sorry, I don't understand your question."

# Test the chatbot
print(chatbot("What are the symptoms of diabetes?"))
print(chatbot("What are the treatments for hypertension?"))
print(chatbot("What is the link between hypertension and diabetes?"))