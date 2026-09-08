# 📄 Data Ingestion Pipeline — `ingest.py`

This module is the **first and most critical step** in the RoleLens retrieval-augmented generation (RAG) pipeline.
Its job is to **load**, **clean**, and **split** raw documents (resumes + job descriptions) into structured chunks that are ready to be embedded and stored in a vector database.

---

## 🗂️ Where It Fits in the Pipeline

```
Raw Files (PDF / TXT)
        │
        ▼
  [ ingest.py ]  ◄── You are here
        │
        ▼
  Cleaned Chunks with Metadata
        │
        ▼
  Embedding Model
        │
        ▼
  Vector Database (e.g. ChromaDB / FAISS)
        │
        ▼
  RAG Query & Retrieval
```

---

## 📁 Data Structure Expected

```
rolelens/
├── data/
│   ├── resumes/
│   │   ├── INFORMATION-TECHNOLOGY/
│   │   │   ├── 12345678.pdf
│   │   │   └── ...
│   │   ├── BANKING/
│   │   ├── SALES/
│   │   └── ... (other categories)
│   └── jobs/
│       ├── 01_accountant.txt
│       ├── 08_banking.txt
│       └── ... (job descriptions)
└── src/
    └── loaders/
        └── ingest.py  ◄── this file
```

---

## ⚙️ How It Works — Step by Step

### Step 1 — Path Setup

```python
BASE_DIR = Path(__file__).resolve().parents[2]
RESUME_DIR = BASE_DIR / "data/resumes"
JOB_DIR    = BASE_DIR / "data/jobs"
```

Uses **absolute paths** derived from the script's own location — not the working directory.  
This means the script works correctly no matter where you run it from.

---

### Step 2 — Text Cleaner: `clean_text()`

```python
def clean_text(text):
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'\xc2|\xe2\x80\x8b', '', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
```

PDFs often produce messy text when parsed. This function handles 4 types of noise:

| Problem | Example | Fix |
|---|---|---|
| Broken UTF-8 encoding | `Â`, `â€‹` | Re-encode and strip bad bytes |
| Hidden control characters | `\x0b`, `\x1f` | Remove with regex |
| Excessive spaces | `Skills:   Python` | Collapse to single space |
| Excessive blank lines | 4+ empty lines | Reduce to max 2 newlines |

Applied **twice**: once when loading each page, and once after splitting each chunk.

---

### Step 3 — Resume Loader: `load_resumes(categories=None)`

```python
def load_resumes(categories=None):
    for pdf_file in RESUME_DIR.glob("**/*.pdf"):
        if categories and pdf_file.parent.name not in categories:
            continue
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()
        ...
```

- Uses `**/*.pdf` to **recursively** find all PDFs across all category subfolders
- `categories` parameter lets you **filter** by job category (e.g. only load `BANKING` resumes) — avoids loading thousands of files at once
- Each page becomes a separate `Document` object with the following metadata:

```python
{
    "doc_type":  "resume",
    "file_name": "12345678.pdf",
    "category":  "INFORMATION-TECHNOLOGY",
    "source":    "/full/path/to/file.pdf",
    "page":      0,              # added automatically by PyPDFLoader
    "total_pages": 2             # added automatically by PyPDFLoader
}
```

- Wrapped in `try/except` to **skip corrupted or unreadable PDFs** without crashing

---

### Step 4 — Job Description Loader: `load_jobs()`

```python
def load_jobs():
    for txt_file in JOB_DIR.glob("*.txt"):
        text = clean_text(txt_file.read_text(encoding="utf-8"))
        docs.append(Document(page_content=text, metadata={...}))
```

- Reads plain `.txt` files directly — no special parser needed
- Each file = one `Document` (job descriptions are short enough to not need page splitting)
- Metadata:

```python
{
    "doc_type":  "job_description",
    "file_name": "08_banking.txt",
    "source":    "/full/path/to/file.txt"
}
```

---

### Step 5 — Chunker: `chunk_documents()`

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)
```

This is the most important configuration. Here's what each setting does:

| Setting | Value | Why |
|---|---|---|
| `chunk_size` | `800` chars | Large enough to hold a full resume section (e.g. work experience entry), small enough for embedding models |
| `chunk_overlap` | `100` chars | Repeats the last 100 chars in the next chunk so context isn't lost at boundaries |
| `separators` | `\n\n → \n → space → ""` | Tries to split at paragraph breaks first, then lines, then words — never cuts mid-sentence if avoidable |

**Visual example of overlap:**
```
Chunk 0: "...managed a team of 6 engineers and led the migration to AWS cloud infrastructure."
                                                              ▲ overlap starts here
Chunk 1: "...led the migration to AWS cloud infrastructure. Reduced costs by 30% in Q3 2022."
```

Each chunk also gets a `chunk_id` for tracking:
```python
chunk.metadata["chunk_id"] = i  # global sequential ID
```

---

## 🚀 Usage

**Load a specific category (recommended for development):**
```python
from ingest import load_resumes, load_jobs, chunk_documents

documents = load_resumes(categories=["INFORMATION-TECHNOLOGY", "BANKING"]) + load_jobs()
chunks = chunk_documents(documents)
```

**Load all resumes (production):**
```python
documents = load_resumes() + load_jobs()
chunks = chunk_documents(documents)
```

**Run directly:**
```bash
python3 ingest.py
```

---

## 📊 Output Example

```
Loaded 271 documents
Created 1349 chunks

page content: INFORMATION TECHNOLOGY MANAGER / NETWORK ENGINEER
Professional Overview
A highly skilled and accomplished Information Technology Manager
with over 16 years of expertise in planning, implementing and
streamlining IT systems...

metadata: {
    'doc_type':    'resume',
    'file_name':   '16533554.pdf',
    'category':    'INFORMATION-TECHNOLOGY',
    'source':      '/home/.../data/resumes/INFORMATION-TECHNOLOGY/16533554.pdf',
    'page':        0,
    'total_pages': 2,
    'chunk_id':    0
}
```

---

## 📦 Dependencies

```
langchain-community       # PyPDFLoader
langchain-core            # Document class
langchain-text-splitters  # RecursiveCharacterTextSplitter
pypdf                     # PDF parsing backend
```

Install:
```bash
pip install langchain-community langchain-core langchain-text-splitters pypdf
```
