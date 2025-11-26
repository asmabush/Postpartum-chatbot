pip install streamlit
import streamlit as st
import json
import random
import nltk
import numpy as np
from tensorflow.keras.models import load_model
import pickle
!pip install pyngrok

# Load the chatbot model and data
lemmatizer = nltk.stem.WordNetLemmatizer()
model = load_model('chatbot_model.h5')
chatbot_data = pickle.load(open('chatbot_data.pkl', 'rb'))

words = chatbot_data['words']
classes = chatbot_data['classes']
intents = chatbot_data['data']['ourIntents']

def clean_input(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha()]
    return tokens

def bag_of_words(sentence, words):
    tokens = clean_input(sentence)
    bag = [0] * len(words)
    for token in tokens:
        for i, w in enumerate(words):
            if w == token:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    result = model.predict(np.array([bow]))[0]
    threshold = 0.6
    if max(result) < threshold:
        return None
    return classes[np.argmax(result)]

def get_response(intent_tag):
    for intent in intents:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn't understand that."


# Streamlit app layout
st.set_page_config(page_title="Postnatal Care Chatbot", page_icon="👶")
st.title('Postnatal Care Chatbot 👶❤️🤱')

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

%%writefile app.py
import streamlit as st
import json
import random
import nltk
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load the chatbot model and data
lemmatizer = nltk.stem.WordNetLemmatizer()
model = load_model('chatbot_model.h5')
chatbot_data = pickle.load(open('chatbot_data.pkl', 'rb'))

words = chatbot_data['words']
classes = chatbot_data['classes']
intents = chatbot_data['data']['ourIntents']

def clean_input(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha()]
    return tokens

def bag_of_words(sentence, words):
    tokens = clean_input(sentence)
    bag = [0] * len(words)
    for token in tokens:
        for i, w in enumerate(words):
            if w == token:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    result = model.predict(np.array([bow]))[0]
    threshold = 0.6
    if max(result) < threshold:
        return 'noanswer' # Return 'noanswer' tag if confidence is below threshold
    return classes[np.argmax(result)]

def get_response(intent_tag):
    for intent in intents:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])
    # This fallback should ideally not be hit if 'noanswer' is handled by predict_class
    return "Sorry, an unexpected error occurred in finding a response." 


# Streamlit app layout
st.set_page_config(page_title="Postnatal Care Chatbot", page_icon="👶")
st.title('Postnatal Care Chatbot 👶❤️🤱')

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

%%writefile -a app.py

# Accept user input
if prompt := st.chat_input("What would you like to know about postnatal care?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    intent_tag = predict_class(prompt)
    # Since predict_class now always returns a tag (or 'noanswer'), no explicit 'else' for fallback is needed here
    response = get_response(intent_tag)

    # Display bot response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

!ngrok authtoken #write your authtoken here


import subprocess

print("Starting Streamlit app in background...")
# Run the Streamlit application in the background
# The &>/dev/null& redirects stdout/stderr to null and runs it in background.
subprocess.Popen(["streamlit", "run", "app.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("Streamlit app started.")

from pyngrok import ngrok

# Terminate any existing ngrok tunnels
ngrok.kill()

# Open a ngrok tunnel to the Streamlit port 8501
public_url = ngrok.connect(addr="8501", proto="http")
print(f"Streamlit App URL: {public_url}")