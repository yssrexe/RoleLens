import re
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parents[2]
RESUME_DIR = BASE_DIR / "data/resumes"
JOB_DIR = BASE_DIR / "data/jobs"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)

def clean_text(text):
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'\xc2|\xe2\x80\x8b', '', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def load_resumes(categories=None):
    docs = []
    for pdf_file in RESUME_DIR.glob("**/*.pdf"):
        if categories and pdf_file.parent.name not in categories:
            continue
        try:
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()
            for page in pages:
                page.page_content = clean_text(page.page_content)
                page.metadata["doc_type"] = "resume"
                page.metadata["file_name"] = pdf_file.name
                page.metadata["category"] = pdf_file.parent.name
                page.metadata["source"] = str(pdf_file)
            docs.extend(pages)
        except Exception as e:
            print(f"Skipping {pdf_file.name}: {e}")
    return docs

def load_jobs():
    docs = []
    for txt_file in JOB_DIR.glob("*.txt"):
        text = clean_text(txt_file.read_text(encoding="utf-8"))
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "doc_type": "job_description",
                    "file_name": txt_file.name,
                    "source": str(txt_file)
                }
            )
        )
    return docs

def chunk_documents(documents):
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.page_content = clean_text(chunk.page_content)
    return chunks

# if __name__ == "__main__":
#     documents = load_resumes(categories=["INFORMATION-TECHNOLOGY"]) + load_jobs()
#     chunks = chunk_documents(documents)

#     print(f"Loaded {len(documents)} documents")
#     print(f"Created {len(chunks)} chunks")
#     print(f"page content: {chunks[0].page_content[:500]}")
#     print(f"metadata: {chunks[0].metadata}")
