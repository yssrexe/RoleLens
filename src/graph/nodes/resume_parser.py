"""Agent 1: extract structured job requirements and resume profiles."""
from src.graph.models import Profile, Requirements

def resume_parser(state, llm):
    requirements = llm.structured(Requirements,
        "Extract explicit mandatory job requirements, excluding optional/nice-to-have skills. "
        "Do not invent requirements. Normalize skill names and education credentials (e.g. bachelor, master, phd). "
        "Use empty lists for unstated skills or education, null for unstated years, "
        "and unknown for unstated seniority. Never put null or placeholder strings in lists.",
        {"job_description": state["job_description"]})
    candidates = []
    for resume in state["selected_resumes"]:
        profile = llm.structured(Profile,
            "Extract resume skills, total non-overlapping years of professional experience, seniority, and education. "
            "Use empty lists for unstated skills or education, null for unstated years, "
            "and unknown for unstated seniority. Never put null or placeholder strings in lists. "
            "Do not infer a degree from a school name. Normalize skill names and education credentials "
            "(e.g. bachelor, master, phd). Do not infer skills from job titles alone.", {"resume": resume["text"]})
        candidates.append({"candidate_id": resume["candidate_id"], "label": resume["label"],
                           "profile": profile.model_dump(), "vector_distance": resume["vector_distance"],
                           "rerank_score": resume["rerank_score"],
                           "retrieved_evidence": resume["retrieved_evidence"]})
    return {"requirements": requirements.model_dump(), "candidates": candidates,
            "stages": [*state["stages"], "resume_parser"]}
