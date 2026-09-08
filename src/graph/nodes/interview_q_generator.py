"""Agent 3: generate interview questions from evidenced gaps."""
from src.graph.models import Questions

def interview_q_generator(state, llm):
    candidates = []
    for candidate in state["candidates"]:
        gaps = candidate["gaps"]
        items = []
        if gaps:
            result = llm.structured(Questions,
                "Generate 1–6 practical interview questions tailored to the candidate and job. "
                "Focus only on the supplied gaps. Copy each question's gap exactly from the gap list. "
                "Include what evidence to listen for; do not invent candidate answers.",
                {"job_description": state["job_description"], "profile": candidate["profile"], "gaps": gaps})
            items = [q.model_dump() for q in result.questions]
            if any(q["gap"] not in gaps for q in items):
                raise ValueError("Question generator returned an unsupported gap")
        candidates.append({**candidate, "questions": items})
    return {"candidates": candidates, "stages": [*state["stages"], "interview_q_generator"]}

