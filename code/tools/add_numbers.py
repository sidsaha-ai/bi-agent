"""
Contains a Tool that adds numnbers.
"""
from langchain.tools import Tool


def add_numbers(numbers: str) -> str:
    """
    Adds the numbers.
    """
    print(f'TOOL NUMBERS: {numbers}')
    try:
        num_list = [int(n) for n in numbers.split(',')]
        print(f'NUMBER LIST: {num_list}')
        res = str(sum(num_list))
        print(f'RESULT: {res}')
        return res
    except Exception as e:
        return f'Exception: {e}'


add_tool = Tool(
    name='AddTool', func=add_numbers, description='This tool adds numbers. Provide comma-separated integers.',
)
