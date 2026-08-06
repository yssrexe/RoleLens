from langgraph.graph import START, END, StateGraph
from typing import TypedDict, Dict

class personized(TypedDict):
    name: str

def personized_test(state: personized):
    state["name"] = "hello Mr " + state["name"] +  "!"
    return state

graph = StateGraph(personized)   
graph.add_node("test", personized_test)
graph.add_edge(START, "test")
graph.add_edge("test", END)
graph = graph.compile()
print(graph)
result = graph.invoke({"name": "Yassir"})

print(result)
