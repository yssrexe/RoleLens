from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Dict


class personized(TypedDict):
    name: str
    age: int
    skills : list[str]
    result : str

def firt_graph(state : personized):
    state['result'] = f"{state['name']}, welcome to the system!"
    return state

def second_graph(state : personized):
    state['result'] += f" you are {state['age']} years old! "
    return state

def third_graph(state : personized):
    
    state['result'] += f" your skills are "
    for skill in state['skills']:
        state['result'] += f" {skill},"
    
    return state    

graph = StateGraph(personized)
graph.add_node("first_graph", firt_graph)
graph.add_node("second_graph", second_graph)
graph.add_node("third_graph", third_graph)

graph.add_edge(START, "first_graph")
graph.add_edge("first_graph", "second_graph")   
graph.add_edge("second_graph", "third_graph")
graph.add_edge("third_graph", END)

graph = graph.compile()
result = graph.invoke({
    "name": "John",
    "age": 30,
    "skills": ["Python", "JavaScript", "SQL"]})

print(result["result"])
