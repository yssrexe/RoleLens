from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

RESUME_DIR = Path("data/resumes")
JOB_DIR = Path("data/jobs")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)

def load_resumes():
    docs = []
    for pdf_file in RESUME_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()
        for page in pages:
            page.metadata["doc_type"] = "resume"
            page.metadata["file_name"] = pdf_file.name
        docs.extend(pages)
    return docs

def load_jobs():
    docs = []
    for txt_file in JOB_DIR.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8")
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
    return chunks

if __name__ == "__main__":
    documents = load_resumes() + load_jobs()
    chunks = chunk_documents(documents)

    print(f"Loaded {len(documents)} documents")
    print(f"Created {len(chunks)} chunks")
    print(chunks[0].page_content[:500])
    print(chunks[0].metadata)