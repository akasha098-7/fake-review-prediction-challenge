import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

def main():
    """Builds and runs a fake review detection system."""
    try:
        # Load the dataset. Make sure your file name matches.
        # The dataset is expected to have a 'text' column and a 'deceptive' column.
        df = pd.read_csv('deceptive_reviews.csv')
    except FileNotFoundError:
        print("Error: 'deceptive_reviews.csv' not found.")
        print("Please download the Deceptive Opinion Spam Corpus and save it in the same directory.")
        return

    # Convert labels to numerical format (0 for 'truthful', 1 for 'deceptive')
    # This line has been updated to use the 'deceptive' column.
    df['label'] = df['deceptive'].apply(lambda x: 1 if x == 'deceptive' else 0)

    # Split the data into features (text) and target (label)
    X = df['text']
    y = df['label']

    # Split the data into a training set and a testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a machine learning pipeline
    # 1. TfidfVectorizer converts text into a matrix of TF-IDF features
    # 2. MultinomialNB is a simple and effective classifier for text data
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model's accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained successfully. Accuracy: {accuracy:.2f}")
    
    # Example of how to use the model to predict a new review
    print("\n--- Example Prediction ---")
    test_review = input("Enter a review to predict if it's fake: ")
    prediction = model.predict([test_review])
    
    if prediction[0] == 1:
        print("Prediction: This review is likely DECEPTIVE.")
    else:
        print("Prediction: This review is likely GENUINE.")

if __name__ == "__main__":
    main()