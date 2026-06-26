from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    """
    print("hello yassir")
    return str(eval(expression))

