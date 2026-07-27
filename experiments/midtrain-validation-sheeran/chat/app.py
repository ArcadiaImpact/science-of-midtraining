"""Streamlit chat UI for an OpenAI-compatible endpoint (remote vLLM via SSH tunnel).

Run: streamlit run app.py
Deps: pip install streamlit httpx
"""
import json
import os

import httpx
import streamlit as st

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("MODEL", "sheeran")
API_KEY = os.environ.get("API_KEY", "EMPTY")
DEFAULT_MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))
DEFAULT_TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))

st.set_page_config(page_title=f"chat · {MODEL}", page_icon="💬")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")
    st.caption(f"Model `{MODEL}` @ `{BASE_URL}`")
    temperature = st.slider("temperature", 0.0, 2.0, DEFAULT_TEMPERATURE, 0.05)
    max_tokens = st.slider("max_tokens", 64, 4096, DEFAULT_MAX_TOKENS, 64)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


def stream_reply(messages, temperature, max_tokens):
    """Yield content chunks from the streaming chat/completions endpoint."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    with httpx.Client(base_url=BASE_URL, timeout=None) as client:
        with client.stream(
            "POST",
            "/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {API_KEY}"},
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: ") :]
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                chunk = delta.get("content")
                if chunk:
                    yield chunk


# Replay history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Message the model…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(
                stream_reply(st.session_state.messages, temperature, max_tokens)
            )
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except httpx.HTTPStatusError as e:
            st.error(f"Server error {e.response.status_code}: {e.response.text}")
            st.session_state.messages.pop()
        except (httpx.ConnectError, httpx.ReadError, httpx.ConnectTimeout):
            st.error(
                f"Can't reach the server — is the SSH tunnel up? Expected {BASE_URL}"
            )
            st.session_state.messages.pop()
