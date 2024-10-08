"""
Contains a Tool that adds numnbers.
"""
from langchain.tools import Tool


def add_numbers(numbers: str) -> str:
    """
    Adds the numbers.
    """
    try:
        num_list = [int(n) for n in numbers.split()]
        return str(sum(num_list))
    except Exception as e:
        return f'Exception: {e}'


add_tool = Tool(
    name='AddTool', func=add_numbers, description='This tool adds numbers. Provide space-separated integers.',
)
