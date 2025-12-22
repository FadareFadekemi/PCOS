import streamlit as st
import requests

# ------------------- 1. Page Configuration -------------------
st.set_page_config(
    page_title="CystaCare Assistant", 
    page_icon="🌸", 
    layout="centered"
)

# ------------------- 2. Custom CSS -------------------
st.markdown("""
<style>
.stApp { background-color: #fffafb; }
section[data-testid="stSidebar"] { background-color: #ffeef2 !important; border-right: 1px solid #f9d5e5; }
div[data-testid="stChatMessage"] { border-radius: 15px; padding: 10px; margin-bottom: 10px; }
.brand-title { font-family: 'Helvetica Neue', sans-serif; color: #d63384; font-weight: 800; font-size: 2.5rem; text-align: center; margin-bottom: 0px; }
.brand-subtitle { color: #6c757d; text-align: center; font-style: italic; margin-bottom: 30px; }
</style>
""", unsafe_allow_html=True)

# ------------------- 3. Sidebar -------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=80)
    st.title("My Account")
    user_id = st.text_input("Username", value="User123", help="Your chats are saved to this ID")
    
    st.divider()
    st.subheader("Saved Consultations")
    
    if st.button("🔄 Refresh History"):
        try:
            resp = requests.get(f"http://localhost:8000/chat-memory", params={"user_id": user_id})
            if resp.status_code == 200:
                st.session_state.messages = resp.json()
                st.toast("History restored!", icon="🌸")
        except:
            st.error("Backend offline")
    
    if st.button("🗑️ Clear Local View"):
        st.session_state.messages = []
        st.rerun()

# ------------------- 4. Main Body -------------------
st.markdown('<p class="brand-title">🌸 CystaCare</p>', unsafe_allow_html=True)
st.markdown('<p class="brand-subtitle">Empowering your journey with PCOS health intelligence</p>', unsafe_allow_html=True)

# ------------------- 5. Initialize Session State -------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------- 6. Display Chat History -------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🌸" if message["role"] == "assistant" else "👤"):
        st.markdown(message["content"])

# ------------------- 7. Chat Input & Streaming -------------------
if prompt := st.chat_input("Ask me about PCOS symptoms, diet, or research..."):
    # Show user message immediately
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Append to session state immediately
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Assistant Response with Streaming
    with st.chat_message("assistant", avatar="🌸"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            payload = {
                "message": prompt,
                "user_id": user_id,
                "app": "CystaCare",
                "history": st.session_state.messages[-6:]  # last 6 messages as context
            }

            with requests.post("http://localhost:8000/chat-stream", json=payload, stream=True) as r:
                for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
            
            # Finalize response
            response_placeholder.markdown(full_response)

            # Save assistant message to local session
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # Save to backend memory
            try:
                requests.post("http://localhost:8000/chat-memory", json={
                    "user_id": user_id,
                    "messages": [
                        {"role": "user", "content": prompt, "source": "user"},
                        {"role": "assistant", "content": full_response, "source": "assistant"}
                    ]
                })
            except:
                st.warning("Could not save chat to memory.")
        
        except Exception as e:
            st.error(f"I'm having trouble connecting to my brain (backend). Is app.py running?\n{e}")
