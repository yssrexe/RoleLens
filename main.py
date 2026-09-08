"""Run the three-agent workflow with text files or existing PGVector retrieval."""
import argparse
import json
from pathlib import Path
from src.graph.workflow import analyze

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True, help="Job description text file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--resumes", nargs="+", type=Path, help="Resume text files")
    group.add_argument("--retrieve", action="store_true", help="Use the existing PGVector retriever (returns chunks)")
    args = parser.parse_args()
    job = args.job.read_text()
    if args.retrieve:
        from src.retrievers.retriever import retrieval_search
        docs = retrieval_search(job, doc_type="resume")
        resumes = [{"label": d.metadata.get("file_name") or d.metadata.get("source") or f"Candidate {i+1}",
                    "text": d.page_content} for i, d in enumerate(docs)]
        if not resumes:
            parser.error("No resumes found in the vector store")
    else:
        resumes = [{"label": p.name, "text": p.read_text()} for p in args.resumes]
    print(json.dumps(analyze({"job_description": job, "resumes": resumes}), indent=2))

if __name__ == "__main__":
    main()
