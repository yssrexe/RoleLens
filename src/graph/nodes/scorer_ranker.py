"""Agent 2: semantic similarity, deterministic fit rules, and ranking."""
import math
from src.graph.models import Profile, Requirements

ALIASES = {"js": "javascript", "ts": "typescript", "postgres": "postgresql",
           "py": "python", "c sharp": "c#", "c++": "c++", "k8s": "kubernetes"}

def normalize(value):
    value = " ".join(value.lower().strip().split())
    return ALIASES.get(value, value)

def score_profile(profile, requirements, semantic):
    if not math.isfinite(semantic):
        raise ValueError("Semantic score must be finite")
    skills = {normalize(s) for s in profile.skills}
    required = {normalize(s) for s in requirements.required_skills if s.strip()}
    matched = sorted(required & skills)
    missing = sorted(required - skills)
    gaps = [f"Skill not evidenced: {skill}" for skill in missing]
    components = {"semantic": max(0., min(1., semantic))}
    weights = {"semantic": .45, "skills": .35, "experience": .10, "seniority": .05, "education": .05}
    if required:
        components["skills"] = len(matched) / len(required)
    if requirements.minimum_years is not None and requirements.minimum_years > 0:
        years = profile.years_experience or 0
        components["experience"] = min(1., years / requirements.minimum_years)
        if years < requirements.minimum_years:
            gaps.append(f"Experience: {profile.years_experience if profile.years_experience is not None else 'unknown'} years evidenced; {requirements.minimum_years:g} required")
    levels = {"unknown": 0, "junior": 1, "mid": 2, "senior": 3, "lead": 4}
    if requirements.seniority != "unknown":
        components["seniority"] = min(1., levels[profile.seniority] / levels[requirements.seniority])
        if components["seniority"] < 1:
            gaps.append(f"Seniority: {profile.seniority} evidenced; {requirements.seniority} required")
    if requirements.education:
        education = {normalize(e) for e in profile.education}
        needed = {normalize(e) for e in requirements.education}
        components["education"] = len(needed & education) / len(needed)
        gaps.extend(f"Education not evidenced: {e}" for e in sorted(needed - education))
    total_weight = sum(weights[key] for key in components)
    return {"fit_score": round(sum(value * weights[key] for key, value in components.items()) / total_weight, 4),
            "score_breakdown": components, "effective_weights": {key: weights[key] / total_weight for key in components},
            "matched_skills": matched, "gaps": gaps}

def scorer_ranker(state, similarity):
    scores = similarity(state["job_description"], [r["text"] for r in state["resumes"]])
    if len(scores) != len(state["candidates"]):
        raise ValueError("Expected one semantic score per candidate")
    requirements = Requirements.model_validate(state["requirements"])
    candidates = [{**c, **score_profile(Profile.model_validate(c["profile"]), requirements, float(score))}
                  for c, score in zip(state["candidates"], scores)]
    candidates.sort(key=lambda c: (-c["fit_score"], c["candidate_id"]))
    return {"candidates": [{**c, "rank": i + 1} for i, c in enumerate(candidates)],
            "stages": [*state["stages"], "scorer_ranker"]}

