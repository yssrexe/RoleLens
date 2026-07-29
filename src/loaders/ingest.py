import re
from pathlib import Path
from pydantic import BaseModel
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

BASE_DIR = Path(__file__).resolve().parents[2]
RESUME_DIR = BASE_DIR / "data/resumes"
JOB_DIR = BASE_DIR / "data/jobs"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)

class ResumeMetadata(BaseModel):
    industry: str
    role: str
    all_skills: List[str]
    years_experience: int
    education_level: str

_prompt = ChatPromptTemplate.from_template(
    "Extract structured information from this resume.\n"
    "Return ONLY a JSON object with keys: industry, role, all_skills (list), "
    "years_experience (int), education_level.\n\nResume:\n{resume_text}"
)
_llm = ChatOllama(model="llama3.2", format="json", temperature=0)
chain = _prompt | _llm.with_structured_output(ResumeMetadata)

def clean_text(text):
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'\xc2|\xe2\x80\x8b', '', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def load_resumes(categories=None):
    docs = []
    numfeild = 0
    for pdf_file in RESUME_DIR.glob("**/*.pdf"):
        if categories and pdf_file.parent.name not in categories:
            continue
        try:
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()

            full_text = " ".join([clean_text(page.page_content) for page in pages])
            try:
                extracted_data = chain.invoke({"resume_text": full_text})
                ai_metadata = extracted_data.model_dump()
            except Exception as e:
                print(f"AI extraction failed for {pdf_file.name}: {e}")
                numfeild += 1
                ai_metadata = {
                    "industry": "Unknown", "role": "Unknown",
                    "all_skills": [], "years_experience": 0,
                    "education_level": "Unknown"
                }

            for page in pages:
                page.page_content = clean_text(page.page_content)
                page.metadata["doc_type"] = "resume"
                page.metadata["file_name"] = pdf_file.name
                page.metadata["category"] = pdf_file.parent.name
                page.metadata["source"] = str(pdf_file)
                page.metadata.update(ai_metadata)
            docs.extend(pages)
        except Exception as e:
            print(f"Skipping {pdf_file.name}: {e}")
    print(f"Number of resumes where AI extraction failed: {numfeild}")
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
