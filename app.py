import streamlit as st
from chatbot import (
    initialize_chatbot,
    retrieve_topk,
    generate_response
)

# ------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Healthcare Chatbot",
    page_icon="🏥",
    layout="wide"
)

# ------------------------------------------------
# LOAD CHATBOT (CACHED)
# ------------------------------------------------
@st.cache_resource
def load_chatbot():
    docs_df, index = initialize_chatbot()
    return docs_df, index

with st.spinner("🔄 Initializing chatbot... Please wait (first load may take a minute)"):
    docs_df, index = load_chatbot()

# ------------------------------------------------
# UI
# ------------------------------------------------
st.title("🏥 Healthcare RAG Chatbot")

st.markdown(
    """
    This chatbot analyzes symptoms and suggests possible related diseases.

    ⚠️ **Disclaimer:** This is *not* a medical diagnosis.
    Always consult a qualified healthcare professional.
    """
)

user_input = st.text_area(
    "🩺 Enter your symptoms (comma separated):",
    placeholder="e.g., fever, sore throat, headache"
)

# ------------------------------------------------
# ACTION
# ------------------------------------------------
if st.button("🔍 Analyze Symptoms"):
    if not user_input.strip():
        st.warning("Please enter at least one symptom.")
    else:
        user_symptoms = [
            s.strip().lower()
            for s in user_input.split(",")
            if s.strip()
        ]

        with st.spinner("Analyzing your symptoms..."):
            retrieved = retrieve_topk(user_symptoms, docs_df, index)
            response = generate_response(user_symptoms, retrieved)

        # ----------------------------
        # RETRIEVED DOCUMENTS
        # ----------------------------
        st.subheader("🧾 Top Retrieved Diseases")

        for doc, score in retrieved:
            with st.expander(f"{doc['disease']} (score: {score:.3f})", expanded=True):
                st.markdown(f"**Description:** {doc['description']}")
                st.markdown(f"**Symptoms:** {', '.join(doc['symptoms'])}")
                st.markdown(f"**Precautions:** {', '.join(doc['precautions'])}")

        # ----------------------------
        # GPT RESPONSE
        # ----------------------------
        st.subheader("💡 AI Health Analysis")
        st.write(response)

        st.markdown("---")
        st.caption(
            "⚠️ This chatbot provides general information only. "
            "Seek professional medical advice for health concerns."
        )
