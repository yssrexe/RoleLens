def calculator(mathstring : str) -> str:
    """
    Evaluate a mathematical expression.
    """
    return str(eval(mathstring))


print(calculator("2 * 2 + 5"))

