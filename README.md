# 🏥 Healthcare 835 RAG Platform

AI-powered healthcare EDI 835 validation and semantic search platform built using Python, Streamlit, ChromaDB, and Sentence Transformers.

This project demonstrates how healthcare companies can:

- Upload external EDI 835 claim files
- Validate claim segments
- Parse structured healthcare transactions
- Store embeddings in vector database
- Perform semantic AI-powered search on claims

---

# 🚀 Features

✅ Upload EDI 835 files  
✅ Validate EDI claim segments  
✅ Parse healthcare claim transactions  
✅ Dashboard view for claims  
✅ Generate embeddings using Sentence Transformers  
✅ Store vectors in ChromaDB  
✅ Semantic AI-powered claim search  
✅ Streamlit UI for easy testing  

---

# 🧠 Architecture

```text
Upload 835 File
      ↓
EDI Validation
      ↓
Claim Parsing
      ↓
Structured Dashboard
      ↓
Embedding Generation
      ↓
ChromaDB Vector Storage
      ↓
Semantic Search
```

---

# 🛠 Tech Stack

- Python 3
- Streamlit
- ChromaDB
- Sentence Transformers
- all-MiniLM-L6-v2 Embedding Model
- Pandas

---

# 📂 Project Structure

```text
healthcare-835-rag-platform/
│
├── app.py
├── sample_835.txt
├── requirements.txt
├── README.md
└── screenshots/
```

---

# ⚙️ Local Setup Instructions

## STEP 1 — Clone Repository

```bash
git clone https://github.com/sangram0864/healthcare-835-rag-platform.git
```

---

## STEP 2 — Go Inside Project

```bash
cd healthcare-835-rag-platform
```

---

## STEP 3 — Create Virtual Environment

### Mac/Linux

```bash
python3 -m venv venv
```

Activate environment:

```bash
source venv/bin/activate
```

---

# 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Application

```bash
streamlit run app.py
```

Application will open automatically in browser.

---

# 📄 Sample EDI 835 File

Create file:

```bash
touch sample_835.txt
```

Open file:

```bash
nano sample_835.txt
```

Add sample content:

```text
CLM|1001|2500|John Doe|Dr Smith|PAID
CLM|1002|1800|Alice Brown|Dr Adams|PENDING
CLM|1003|3200|Robert King|Dr Lee|PAID
INVALID|BAD|DATA
```

Save:
- CTRL + O
- Enter
- CTRL + X

---


STEP 7 — ADD REQUIREMENTS FILE

Open:

nano requirements.txt

Add:

streamlit
pandas
chromadb
sentence-transformers

Save.

STEP 8 — RUN APPLICATION
streamlit run app.py


# 🔍 Example Semantic Queries

```text
Show paid claims for John Doe
```

```text
Show pending claims
```

```text
Find claims related to Dr Smith
```

---

# 🤖 AI Workflow

```text
Claim Data
    ↓
Embedding Model
    ↓
Vector Embeddings
    ↓
ChromaDB
    ↓
Semantic Retrieval
```

---

# 📸 Future Improvements

- Persistent ChromaDB storage
- Real EDI 835 parser integration
- OpenAI/Claude integration
- FastAPI backend APIs
- Claim analytics dashboard
- Agentic AI workflows
- PDF export
- Role-based authentication

---

# 👨‍💻 Author

Sangram Desai

GitHub:
https://github.com/sangram0864