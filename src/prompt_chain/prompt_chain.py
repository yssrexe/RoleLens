
from langchain_ollama import ChatOllama

from src.config import OLLAMA_MODEL


llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)


def prepare_dict(result):
    candidate_dict = {
        "file_name": result.metadata.get("file_name", "unknown"),
        "role": result.metadata.get("role", "unknown"),
        "all_skills": result.metadata.get("all_skills", []),
        "years_experience": result.metadata.get("years_experience", "unknown"),
        "education_level": result.metadata.get("education_level", "unknown"),
        "category": result.metadata.get("category", "unknown"),
        "rerank_score": result.metadata.get("rerank_score", "unknown"),
    }
    return candidate_dict


def prompt_template(candidate_dict, query):
    prompt = f"""
You are a recruiter reviewing one resume against a role requirement.

Role requirement:
{query}

Resume information:
- Name: {candidate_dict['file_name']}
- Role: {candidate_dict['role']}
- Skills: {candidate_dict['all_skills']}
- Years of Experience: {candidate_dict['years_experience']}
- Education Level: {candidate_dict['education_level']}
- Category: {candidate_dict['category']}
- Cross-encoder score: {candidate_dict['rerank_score']}

Write the result in exactly this format for one candidate only:

========================================
Name   : <{candidate_dict['file_name']}>
Score  : <one decimal number out of 10>
Strengths : <short sentence based only on the resume>
Gaps      : <short sentence about what is missing relative to the role>
Questions :
    <skill 1>:
        1. <question>
        1. <response>
        2. <question>
        2. <response>
    <skill 2>:
        1. <question>
        1. <response>
        2. <question>
        2. <response>

Generate questions for every skill in the resume's all_skills list.
Each skill must have exactly 2 distinct interview questions and also under every question add a simple resonse.
Questions should be specific, practical, and relevant to the role requirement.

Rules:
- Use only information supported by the resume and the role requirement.
- Keep Strengths and Gaps to one concise sentence each.
- Keep each question short and focused.
- Do not add extra commentary before or after the block.
"""
    return prompt


def generate_questions_for_resumes(results, query):
    generated_outputs = []
    for result in results:
        candidate_dict = prepare_dict(result)
        prompt = prompt_template(candidate_dict, query)
        response = llm.invoke(prompt)
        generated_outputs.append(
            {
                "file_name": candidate_dict["file_name"],
                "prompt": prompt,
                "response": response.content if hasattr(response, "content") else str(response),
            }
        )
    return generated_outputs

