#depression v normal NLP project 
import numpy as np 
import pandas as pd 
import nltk
import tensorflow as tf
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
# test commit -4th commit

df = pd.read_csv("training_dataset.csv")
#df = df.iloc[:,1:]
#print(df.head())


STOP_WORDS = set(stopwords.words('english'))
# define a function that will clean the text 
# this will lower case all the words then remove all special characters 
# then the cleaned text will be tokenized and the stop words will be removed 
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in STOP_WORDS]
    return ' '.join(tokens)

def prepare_data(texts, labels):
    """Prepare data for model training"""
    # Preprocess texts
    processed_texts = [clean_text(text) for text in texts]
    
    # Create vectorizer and label encoder
    vectorizer = TfidfVectorizer(max_features=5000)
    label_encoder = LabelEncoder()
    
    # Vectorize texts
    X = vectorizer.fit_transform(processed_texts).toarray()
    y = label_encoder.fit_transform(labels)
    
    return X, y, vectorizer, label_encoder

def build_model(input_shape, num_classes):
    """Create sentiment analysis neural network"""
    model = tf.keras.Sequential([
        # Reshape input for 1D convolution
        tf.keras.layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
        # Convolutional layers
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dropout(0.3),
        # Dense layers
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='relu'),
        
        # Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_sentiment_model(texts, labels, test_size=0.2, epochs=5, batch_size=32):
    """Train sentiment analysis model"""
    # Prepare data
    X, y, vectorizer, label_encoder = prepare_data(texts, labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Build model
    model = build_model(X.shape[1], len(label_encoder.classes_))
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size
    )
    
    return model, vectorizer, label_encoder

def predict_sentiment(model, vectorizer, label_encoder, texts):
    """Predict sentiment for given texts"""
    # Preprocess and vectorize
    processed_texts = [clean_text(text) for text in texts]
    X = vectorizer.transform(processed_texts).toarray()
    
    # Predict
    predictions = model.predict(X)
    predicted_labels = label_encoder.inverse_transform(
        np.argmax(predictions, axis=1)
    )
    
    return predicted_labels

def main():
    df= pd.read_csv('training_dataset.csv')
    df = df.sample(frac =1 , random_state = 42).reset_index(drop=True)
    df.dropna(inplace=True)
    texts = df['statement'].values
    labels = df['status'].values
    model, vectorizer, label_encoder = train_sentiment_model(texts, labels)

    df2=pd.read_csv('testing_dataset.csv')
    df2 = df2.sample(frac =1 , random_state = 42).reset_index(drop=True)
    df2.dropna(inplace=True)
    new_labels = df2['status'].values
    new_texts = df2['statement'].values
    predictions = predict_sentiment(model, vectorizer, label_encoder, new_texts)
    
    for text, prediction, actual in zip(new_texts, predictions, new_labels):
        print(f"Text: {text[:100]}...")  # Truncate long texts
        print(f"Predicted Sentiment: {prediction}")
        print(f"Actual Sentiment:    {actual}")
        print("-" * 50)
    accuracy = np.mean(predictions == new_labels)
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()

