import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import LancasterStemmer
from tensorflow.keras.models import load_model
import streamlit as st

stemmer = LancasterStemmer()

model = load_model("model.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    threshold = 0.5
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if not intents_list:
        return "🤖 I'm sorry, I don't have information about that. Please try asking something else."
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "🤖 I'm sorry, I didn't understand that."

st.set_page_config(page_title="First Aid Chatbot", page_icon="🩺")
st.title("🩹 First Aid Chatbot")

st.sidebar.title("ℹ️ About the Chatbot")
st.sidebar.markdown("""
This **First Aid Chatbot** helps you get quick guidance on basic first aid situations.

### ⚠️ Disclaimer
This tool is for informational purposes only.  Always seek professional medical help in emergencies.
""")

USER_ICON = "https://cdn-icons-png.flaticon.com/512/147/147144.png"
BOT_ICON = "https://cdn-icons-png.flaticon.com/512/4712/4712027.png"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Type a first aid question...")

if user_input:
    st.session_state.chat_history.append({"message": user_input, "is_user": True})
    
    st.markdown(f"""
    <div style="display:flex; align-items:center; margin-bottom:10px;">
        <img src="{USER_ICON}" style="width:40px; height:40px; border-radius:50%; margin-right:10px;">
        <div style="background-color:#4CAF50; color:white; padding:10px; border-radius:10px; max-width:70%;">
            <b>You:</b> {user_input}
        </div>
    </div>
    """, unsafe_allow_html=True)

    typing_placeholder = st.empty()
    typing_placeholder.markdown(f"""
    <div style="display:flex; align-items:center; margin-bottom:10px; justify-content:flex-end;">
        <div style="background-color:#B0BEC5; color:white; padding:10px; border-radius:10px; font-style:italic; max-width:70%;">
            🤖 Bot is typing...
        </div>
        <img src="{BOT_ICON}" style="width:40px; height:40px; border-radius:50%; margin-left:10px;">
    </div>
    """, unsafe_allow_html=True)

    import time
    time.sleep(1.5)

    intents_result = predict_class(user_input)
    response = get_response(intents_result, intents)

    typing_placeholder.empty()

    st.session_state.chat_history.append({"message": response, "is_user": False})

if not st.session_state.chat_history:
    st.markdown("""
    <div style="text-align:center; margin-top:50px; color:gray;">
        🤖 <b>Welcome to the First Aid Chatbot!</b><br>
        Ask me questions like:<br>
        - What to do in case of a burn? 🔥<br>
        - How do I treat a sprain? 🦶<br>
        - Someone is choking, what do I do? 😨<br><br>
        Type your question below to get started.
    </div>
    """, unsafe_allow_html=True)

for chat in st.session_state.chat_history[1:]:
    if chat['is_user']:
        st.markdown(f"""
        <div style="display:flex; align-items:center; margin-bottom:10px;">
            <img src="{USER_ICON}" style="width:40px; height:40px; border-radius:50%; margin-right:10px;">
            <div style="background-color:#4CAF50; color:white; padding:10px; border-radius:10px; max-width:70%;">
                <b>You:</b> {chat['message']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display:flex; align-items:center; margin-bottom:10px; justify-content:flex-end;">
            <div style="background-color:#607D8B; color:white; padding:10px; border-radius:10px; max-width:70%;">
                <b>Bot:</b> {chat['message']}
            </div>
            <img src="{BOT_ICON}" style="width:40px; height:40px; border-radius:50%; margin-left:10px;">
        </div>
        """, unsafe_allow_html=True)
