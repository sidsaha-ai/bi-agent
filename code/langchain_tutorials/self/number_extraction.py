"""
This script uses Langchain to extract numbers from a user message using
LLM.
"""
import argparse

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def main(user_message: str) -> None:
    """
    The main method where execution starts.
    """
    base_url: str = 'http://localhost:1234/v1'
    api_key: str = 'lm_studio'

    # make the prompt
    system_prompt: str = '''
    Extract the numbers from the question. Just provide the numbers or say None. Below are a few examples -
    # Example 1
    Question: Tell me the sum of 10, 12, and 14.
    AI: 10, 12, 14

    # Example 2
    Question: Can you get me the sum of 1 and 5.
    AI: 1, 5

    # Example 3
    Question: My name is Siddharth.
    AI: None

    # Example 4
    Question: What is the sum of Sid.
    AI: None
    '''
    user_prompt: str = f'''
    Below is the question you have to work on -
    {user_message}
    '''
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', '{system_prompt}'),
        ('user', '{user_prompt}'),
    ])

    # make the model
    model = ChatOpenAI(
        temperature=0.0, base_url=base_url, api_key=api_key,
    )

    # make the output parser
    parser = StrOutputParser()

    # make the chain
    chain = prompt_template | model | parser

    data = {
        'system_prompt': system_prompt,
        'user_prompt': user_prompt,
    }
    result = chain.invoke(data)

    print(result)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--user_message', required=True, type=str,
    )
    args = arg_parser.parse_args()

    main(args.user_message)
