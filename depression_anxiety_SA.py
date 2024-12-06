#sentiment analysis test 

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

STOP_WORDS = set(stopwords.words('english'))

def clean_text(text):
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
        tf.keras.layers.Dropout(0.4),
        
        # Dense layers
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
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

def train_sentiment_model(texts, labels, test_size=0.2, epochs=250, batch_size=32):
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
    # Sample dataset
    texts = [
        "oh my gosh",
        "trouble sleeping, confused mind, restless heart. All out of tune",
        "All wrong, back off dear, foward doubt. Stay in a restless and restless place",
        "I've shifted my focus to something else but i'm still worried",
        "I'm restless and restless, it's been a month now, boy. What do you mean?" , 
        "Suggest a song that makes you calm dont know why suddently this feeling of anger/ restlessness a...",
        "teroooosss nervous huuu",
        "im broken and fragile im hurt i cant stand the anxiety anymore I cant stand the love . until n...",
        "its a habit from elementary school if I think about something ill get worried I cant sleep" , 
        "since yesterday yesterday I have not been in the mood very sensitive feeling restless always sur...", 
        "tonight he continues to chatter about money, Im so confnused what to do once he said dont worry ab...",
        "Did you ever read it when you red it to RL , I mean the atmosphere the taste its because im like ...",
        "since the pandemic I havent opened my laptop for a day, I feel restless I have a very heavy Life", 
        "People who are restless and afraid cn be seen from the frequent touching of the face and ringing th...",
        "Sensitive feelings make the heart restless",

        "I recently went through a breakup and she said she still wants to be friends so I said I can try doi...",
        "I do not know how to navigate these feelings, not that its a new feeling by any stretch. I just do n....",
        "so I have been with my bf for 5 months and he arlready told me he was deoressed. To this week nothing....",
        "I have been severly bullied since I was 5 till 15, this resulted in me being depressed misanthrope",
        "My mom made me go to a camp that she knows I hate now I hate most days the only good time is at mid...",
        "I have not seen my 7 year old daughre in a couple of months because she moved aacross the country wi...",
        "I cannot seem to go a couple of months without self-sabotaging myself . I do not know what comes firs...",
        "I cannot fucking feel a single fucking thing man I bottle up every feeling and I am so far away fr...",
        "cut onions so that I could get some tears out, since I cannot seem to cry and she would single tea...",
        "I am only 21 but everyone around me just tells me that I always act so serious and sometimes even d...", 
        "I no longer look foward to anything I have a beautiful girlfriend, a caring family, job ...",
        "I have just been sitting onn my couch and I am just feeling lost about what do about my dog becaus...",
        "It always gets worseI have no friends, nobody can stand me I have horrible acne and I look absolute...",
        "I am so damn exhasuted of my mind screwung everything up . I am about to turn 30 in a few weeks an a ...",
        "After this pandemic is over and I can finally go out, I will overdose myself will sleeping pills to...",
        "I feel like a burden to everyone including myself , In my life, one that no one wants to help . Feeli...",
        "Anyone feel like making friends in your 20s after university is pretty much impossible? I feel...", 
        "This has been my life for years now. just stay alive until the next day, and then the day after that ..."

    ]
    labels = ['anxiety', 'anxiety', 'anxiety', 'anxiety', 'anxiety' , 'anxiety' , 'anxiety' , 'anxiety' ,'anxiety','anxiety','anxiety','anxiety','anxiety','anxiety','anxiety',
               'depression','depression','depression','depression','depression','depression','depression','depression','depression','depression','depression','depression','depression','depression','depression'
               , 'depression','depression','depression' ]

    # Train model
    model, vectorizer, label_encoder = train_sentiment_model(texts, labels)

    # Predict sentiment
    new_texts = [
        "I hate myself I fucking suck I am the most unstable fucker alive",
        "If the guy is sick, the girl acutally feels sick inndirectly .He must be feeling restless and restles..."
    ]
    predictions = predict_sentiment(model, vectorizer, label_encoder, new_texts)
    
    # Print results
    for text, prediction in zip(new_texts, predictions):
        print(f"Text: {text}")
        print(f"Sentiment: {prediction}")

if __name__ == "__main__":
    main()
