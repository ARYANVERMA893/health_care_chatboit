# health_care_chatbot

This project is a Healthcare Retrieval-Augmented Generation (RAG) Chatbot that analyzes user-provided symptoms and suggests possible related diseases along with brief reasoning, descriptions, and precautionary measures. The system combines semantic search and large language model reasoning to deliver cautious, explainable health information while clearly stating that it is not a medical diagnosis. Users interact with the chatbot through a Streamlit web interface, where symptoms are entered in natural language and processed in real time.

Internally, the chatbot uses Sentence-Transformers (all-MiniLM-L6-v2) to convert symptom queries and medical documents into vector embeddings. These embeddings are indexed using FAISS, enabling fast similarity search to retrieve the most relevant disease records from a structured medical dataset. The retrieved context—including disease names, associated symptoms, descriptions, and precautions—is then passed to an OpenAI GPT model, which generates a concise, human-readable medical explanation following a safety-focused prompt.

The medical knowledge base is built from multiple CSV files that map diseases to symptoms, descriptions, and precautionary steps. To improve performance, the FAISS index and embeddings are cached and reused across application restarts. The chatbot also supports a command-line mode for terminal interaction, in addition to the Streamlit UI. Overall, this project demonstrates a practical application of RAG architecture in healthcare, integrating vector databases, transformer embeddings, and LLMs to provide informative, ethical, and user-friendly health guidance.

Project Structure

healthcare-chatbot/
│
├── app.py                 # Streamlit application
├── chatbot.py             # RAG logic (FAISS + GPT)
├── faiss.index            # Saved FAISS index
├── faiss.index.emb.npy    # Embeddings cache
│
├── dataset/
│   └── archive/
│       ├── dataset.csv
│       ├── symptom_Description.csv
│       ├── symptom_precaution.csv
│       └── Symptom-severity.csv
│
├── requirements.txt
└── README.md
