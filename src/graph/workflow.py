"""Three specialized agents coordinated by a typed LangGraph StateGraph."""
from functools import partial
from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from src.graph.models import AnalysisRequest
from src.graph.nodes.resume_parser import resume_parser
from src.graph.nodes.scorer_ranker import scorer_ranker, score_profile
from src.graph.nodes.interview_q_generator import interview_q_generator
from src.graph.services import OllamaService, semantic_scores

class State(TypedDict, total=False):
    job_description: str
    resumes: list[dict]
    requirements: dict
    candidates: list[dict]
    stages: list[str]

def build_workflow(llm=None, similarity=None):
    llm = llm if llm is not None else OllamaService()
    similarity = similarity if similarity is not None else semantic_scores

    graph = StateGraph(State)
    graph.add_node("resume_parser", partial(resume_parser, llm=llm))
    graph.add_node("scorer_ranker", partial(scorer_ranker, similarity=similarity))
    graph.add_node("interview_q_generator", partial(interview_q_generator, llm=llm))
    graph.add_edge(START, "resume_parser")
    graph.add_edge("resume_parser", "scorer_ranker")
    graph.add_edge("scorer_ranker", "interview_q_generator")
    graph.add_edge("interview_q_generator", END)
    return graph.compile()

def analyze(payload, workflow=None):
    request = AnalysisRequest.model_validate(payload)
    if not request.job_description.strip() or any(not r.text.strip() or not r.label.strip() for r in request.resumes):
        raise ValueError("Job description, resume text and labels must not be blank")
    result = (workflow if workflow is not None else build_workflow()).invoke(request.model_dump())
    return {key: result[key] for key in ("requirements", "candidates", "stages")}
