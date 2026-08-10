from langgraph.graph import StateGraph, START, END
from langchain_core.runnables.graph import MermaidDrawMethod
from typing import TypedDict, Dict
from pathlib import Path


class Calculate(TypedDict):
    a: int
    operation: str
    b: int
    c: int
    d: int
    operation2: str
    result: int
    result2: int

def add(state: Calculate):
    return {'result': state['a'] + state['b']}

def subtract(state: Calculate):
    return {'result2': state['c'] - state['d']}

def route1(state: Calculate):
    if state['operation'] == "+":
        return "add_option"
    elif state['operation'] == "-":
        return "subtract_option"

def route2(state: Calculate):
    if state['operation2'] == "+":
        return "add_option2"
    elif state['operation2'] == "-":
        return "subtract_option2"

graph = StateGraph(Calculate)
graph.add_node("add", add)
graph.add_node("subtract", subtract)
graph.add_node("route1", lambda state: {})
graph.add_node("route2", lambda state: {})



graph.add_edge(START, "route1")
graph.add_conditional_edges("route1", route1,
                            {
                                "add_option": "add",
                                "subtract_option": "subtract"
                            })
graph.add_edge("route1", "route2")
graph.add_conditional_edges("route2", route2,
                            {
                                "add_option2": "add",
                                "subtract_option2": "subtract"
                            })
graph.add_edge("add", END)
graph.add_edge("subtract", END)

graph = graph.compile()
result = graph.invoke({
    "a": 5,
    "operation": "+",
    "b": 3,
    "c": 10,
    "d": 3,
    "operation2": "-"})

print(result["result"])  # Output: 8
print(result["result2"])  # Output: 7