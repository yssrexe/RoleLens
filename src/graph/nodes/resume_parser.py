"""Agent 1: extract structured job requirements and resume profiles."""
from src.graph.models import Profile, Requirements

def resume_parser(state, llm):
    requirements = llm.structured(Requirements,
        "Extract explicit mandatory job requirements, excluding optional/nice-to-have skills. "
        "Do not invent requirements. Normalize skill names and education credentials (e.g. bachelor, master, phd). "
        "Use null for unstated years and unknown for unstated seniority.",
        {"job_description": state["job_description"]})
    candidates = []
    for index, resume in enumerate(state["resumes"]):
        profile = llm.structured(Profile,
            "Extract resume skills, total non-overlapping years of professional experience, seniority, and education. "
            "Use null/unknown when unsupported. Normalize skill names and education credentials "
            "(e.g. bachelor, master, phd). Do not infer skills from job titles alone.", {"resume": resume["text"]})
        candidates.append({"candidate_id": index + 1, "label": resume["label"], "profile": profile.model_dump()})
    return {"requirements": requirements.model_dump(), "candidates": candidates, "stages": ["resume_parser"]}

