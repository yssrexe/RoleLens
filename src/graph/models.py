from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

class StructuredModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

class Profile(StructuredModel):
    skills: list[str] = Field(default_factory=list)
    years_experience: float | None = Field(default=None, ge=0, le=80)
    seniority: Literal["junior", "mid", "senior", "lead", "unknown"] = "unknown"
    education: list[str] = Field(default_factory=list)

class Requirements(StructuredModel):
    required_skills: list[str] = Field(default_factory=list)
    minimum_years: float | None = Field(default=None, ge=0, le=80)
    seniority: Literal["junior", "mid", "senior", "lead", "unknown"] = "unknown"
    education: list[str] = Field(default_factory=list)

class Question(StructuredModel):
    gap: str
    question: str
    what_to_listen_for: str

class Questions(StructuredModel):
    questions: list[Question] = Field(min_length=1, max_length=6)

class ResumeInput(StructuredModel):
    label: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=20, max_length=50000)

class AnalysisRequest(StructuredModel):
    job_description: str = Field(min_length=20, max_length=20000)
    resumes: list[ResumeInput] = Field(min_length=1, max_length=10)
