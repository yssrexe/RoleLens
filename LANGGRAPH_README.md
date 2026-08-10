# LangGraph Integration Plan

This project already has the pieces needed for a LangGraph workflow. The goal is to turn the current linear flow into a three-agent pipeline:

1. Resume Parser - extract structured candidate fields
2. Scorer/Ranker - compute a candidate-job fit score from semantic and rule-based signals
3. Interview Q Generator - create tailored interview questions from the candidate gaps

## What to build first

Start by keeping the existing retrieval and generation code, then move them behind LangGraph nodes one by one. Do not try to redesign the whole app at once.

The best entry point is the orchestration layer:

- `main.py` should eventually only load input and call the graph
- a new graph module should own state, routing, and node execution
- the current retrieval and prompt code should be reused inside graph nodes

## Suggested file layout

Create a small graph-focused layer under `src/`:

- `src/graph/state.py` - shared graph state definition
- `src/graph/nodes/resume_parser.py` - Agent 1
- `src/graph/nodes/scorer_ranker.py` - Agent 2
- `src/graph/nodes/interview_q_generator.py` - Agent 3
- `src/graph/workflow.py` - LangGraph wiring and edges
- `main.py` - demo entrypoint that calls the graph

You can keep the existing files for reuse:

- `src/retrievers/retriever.py` - semantic candidate retrieval
- `src/prompt_chain/prompt_chain.py` - question generation logic, later moved into a node
- `src/config.py` - model names and top-k settings

## Agent responsibilities

### Agent 1 - Resume Parser

Goal: convert raw resume text or retrieved resume chunks into structured fields.

Output shape example:

- `candidate_name`
- `skills`
- `years_experience`
- `seniority`
- `education_level`
- `target_role`
- `source_resume_text`

Implementation notes:

- use a lightweight extraction prompt or an LLM structured output schema
- if metadata already exists in your loaded documents, prefer that first
- keep this node focused on normalization, not ranking

### Agent 2 - Scorer/Ranker

Goal: score the candidate against the job requirement with both semantic and rule-based logic.

Recommended score breakdown:

- semantic similarity from embeddings or retrieval score
- skill overlap score from parsed skills vs job requirements
- seniority fit score from years of experience and role level
- education fit score if the role requires it

Final output example:

- `fit_score` from `0.0` to `1.0`
- `score_breakdown` with component scores
- `match_reasons`
- `gap_reasons`

Implementation notes:

- use deterministic rules for the rule-based part
- keep the final scoring formula visible and easy to tune
- let LangGraph decide whether to continue to the question generator based on score thresholds

### Agent 3 - Interview Q Generator

Goal: generate interview questions from the gaps found by the scorer.

Input should include:

- parsed candidate profile
- job requirement
- gap list from the scorer
- final fit score

Output example:

- `questions_by_skill`
- `questions_by_gap`
- `follow_up_questions`

Implementation notes:

- generate questions only from missing or weak areas
- keep questions practical and role-specific
- reuse the current prompt logic as a starting point, but feed it structured gaps instead of raw resume chunks

## Recommended graph flow

Use a simple pipeline first:

1. Input job description or role query
2. Retrieve candidate resumes
3. Parse each resume into structured fields
4. Score each candidate against the job
5. If score is high enough, generate questions
6. Return the final ranked output

You can later add branching like:

- skip question generation for low-quality candidates
- send ambiguous resumes through a second parsing pass
- rerun ranking with different thresholds

## Step-by-step implementation order

### Step 1 - Define graph state

Create a shared state object that carries the data between nodes.

Include fields such as:

- `job_query`
- `job_description`
- `retrieved_docs`
- `parsed_candidates`
- `scores`
- `gaps`
- `interview_questions`

### Step 2 - Build Agent 1

Wrap resume parsing into one node.

The node should:

- accept retrieved resume content
- extract or normalize skills, years, education, and seniority
- return structured candidate objects

### Step 3 - Build Agent 2

Implement the scorer/ranker as a separate node.

The node should:

- compare the parsed candidate profile with the job requirement
- compute a `0-1` fit score
- store the reasons for the score so they can be reused later

### Step 4 - Build Agent 3

Implement the interview question generator as the final node.

The node should:

- read the gap list from the scorer
- create tailored interview questions for each gap
- optionally group them by skill or competency

### Step 5 - Wire the graph

Create the LangGraph workflow with clear edges between nodes.

Start with a straight line:

- `retrieve -> parse -> score -> questions -> output`

Then add conditional routing if needed.

### Step 6 - Replace the current entrypoint

Update `main.py` so it calls the LangGraph workflow instead of calling retrieval and prompt generation directly.

### Step 7 - Add tests

Add small tests for each node and for the full graph flow.

Focus on:

- parser output shape
- scoring formula behavior
- question generation from gaps
- routing decisions in the graph

## Practical scoring idea

A simple first version can combine these signals:

```text
fit_score = 0.45 * semantic_score + 0.35 * skill_overlap + 0.15 * seniority_fit + 0.05 * education_fit
```

Keep the weights configurable in `src/config.py` so you can tune them later.

## Where to start in the current codebase

The most useful starting points are:

- [main.py](main.py)
- [src/retrievers/retriever.py](src/retrievers/retriever.py)
- [src/prompt_chain/prompt_chain.py](src/prompt_chain/prompt_chain.py)
- [src/config.py](src/config.py)

Those files already contain the flow you want to split into LangGraph nodes.

## Good first milestone

A good first milestone is:

- keep retrieval exactly as it is
- add one parser node that converts a resume into structured data
- add one scorer node that produces a `0-1` fit score
- print the score from `main.py`

Once that works, the interview question generator is straightforward to add.