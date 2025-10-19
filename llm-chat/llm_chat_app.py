import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ----------------------------------------------------------
# Streamlit page setup
# ----------------------------------------------------------
st.set_page_config(page_title="Generative AI Chat")

st.title("Generative AI Chat Assistant")
st.write(
    "This Streamlit app connects to an OpenAI-compatible API. "
    "It demonstrates how to integrate a generative AI model into a simple "
    "interactive chat interface for reasoning, explanation, and prediction."
)

# ----------------------------------------------------------
# Environment and API setup
# ----------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not api_key:
    st.error("API key not found. Please set OPENAI_API_KEY in your .env file.")
    st.stop()

client = OpenAI(api_key=api_key)

# ----------------------------------------------------------
# Session state initialization
# ----------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant that explains reasoning clearly."}
    ]

# ----------------------------------------------------------
# Display previous chat history
# ----------------------------------------------------------
for msg in st.session_state["messages"]:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ----------------------------------------------------------
# User input and response handling
# ----------------------------------------------------------
user_prompt = st.chat_input("Ask a question or start a conversation...")
if user_prompt:
    # Display user input
    st.session_state["messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Generate model response
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            response = client.chat.completions.create(
                model=model_name,
                messages=st.session_state["messages"],
                temperature=0.7,
            )
            reply = response.choices[0].message.content
            st.markdown(reply)

    # Save assistant reply
    st.session_state["messages"].append({"role": "assistant", "content": reply})

st.caption("Built with Streamlit and OpenAI API | Author: Karan Anand")
