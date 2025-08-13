import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("next_word_lstm.h5")

# Load tokenizer (from training phase)
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

total_words = len(tokenizer.word_index) + 1
max_sequence_len = model.input_shape[1] + 1  # +1 because input shape excludes label

def preprocess_input(text):
    """
    Convert raw text to padded sequence, 
    ensuring all tokens are within vocabulary.
    """
    words = text.lower().split()

    # Map words to tokens, use OOV index (1) for unknown words
    token_list = [tokenizer.word_index.get(word, 1) for word in words]

    # Pad to match training sequence length
    padded = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    return padded

# Streamlit UI
st.title('Next Word Prediction (LSTM)')
user_input = st.text_input("Enter your text:")

if st.button("Predict Next Word"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        padded_sequence = preprocess_input(user_input)

        # Predict probabilities
        predicted_probs = model.predict(padded_sequence, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=-1)[0]

        # Convert index back to word
        predicted_word = None
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                predicted_word = word
                break

        if predicted_word:
            st.success(f"Next word suggestion: **{predicted_word}**")
        else:
            st.warning("Could not find prediction in vocabulary.")
