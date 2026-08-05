import os
import uuid

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Omatsuri", page_icon="🎏", layout="centered")
st.title("🎏 Omatsuri")
st.caption("Ask me anything about Japanese matsuri (festivals)")

# --- session state init ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ratings" not in st.session_state:
    st.session_state.ratings = {}  # index → 1 or -1

# --- sidebar ---
with st.sidebar:
    st.header("Settings")
    search_mode = st.selectbox(
        "Search mode",
        options=["hybrid", "dense", "lexical"],
        index=0,
        help="hybrid = BM25 + dense vectors (best); dense = vector only; lexical = BM25 only",
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.ratings = {}
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.caption(f"Session: `{st.session_state.session_id[:8]}…`")


def send_feedback(msg_idx: int, rating: int) -> None:
    msg = st.session_state.messages[msg_idx]
    try:
        requests.post(
            f"{BACKEND_URL}/feedback",
            json={
                "session_id": st.session_state.session_id,
                "query": msg["query"],
                "answer": msg["answer"],
                "search_mode": msg["search_mode"],
                "rating": rating,
            },
            timeout=5,
        )
    except Exception:
        pass
    st.session_state.ratings[msg_idx] = rating


# --- render chat history ---
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message("user"):
        st.write(msg["query"])

    with st.chat_message("assistant"):
        st.write(msg["answer"])

        with st.expander(f"Sources ({len(msg['retrieved'])} records, {msg['latency_ms']:.0f} ms)"):
            for rec in msg["retrieved"]:
                st.markdown(
                    f"**{rec['name']}** · {rec['City']}, {rec['Prefecture']} · {rec['Date']}  \n"
                    f"{rec['Description']}"
                )

        if i in st.session_state.ratings:
            rating = st.session_state.ratings[i]
            st.caption("👍 Thanks!" if rating == 1 else "👎 Noted, thanks!")
        else:
            col1, col2, col3 = st.columns([1, 1, 8])
            with col1:
                if st.button("👍", key=f"up_{i}"):
                    send_feedback(i, 1)
                    st.rerun()
            with col2:
                if st.button("👎", key=f"down_{i}"):
                    send_feedback(i, -1)
                    st.rerun()


# --- chat input ---
if prompt := st.chat_input("Ask about a matsuri…"):
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching…"):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={
                        "query": prompt,
                        "search_mode": search_mode,
                        "session_id": st.session_state.session_id,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                st.error(f"Backend error: {e}")
                st.stop()

        st.write(data["answer"])

        with st.expander(f"Sources ({len(data['retrieved'])} records, {data['latency_ms']:.0f} ms)"):
            for rec in data["retrieved"]:
                st.markdown(
                    f"**{rec['name']}** · {rec['City']}, {rec['Prefecture']} · {rec['Date']}  \n"
                    f"{rec['Description']}"
                )

        msg_idx = len(st.session_state.messages)
        st.session_state.messages.append(
            {
                "query": prompt,
                "answer": data["answer"],
                "retrieved": data["retrieved"],
                "search_mode": data["search_mode"],
                "latency_ms": data["latency_ms"],
            }
        )

        col1, col2, col3 = st.columns([1, 1, 8])
        with col1:
            if st.button("👍", key=f"up_{msg_idx}"):
                send_feedback(msg_idx, 1)
                st.rerun()
        with col2:
            if st.button("👎", key=f"down_{msg_idx}"):
                send_feedback(msg_idx, -1)
                st.rerun()
