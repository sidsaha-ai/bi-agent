"""
This script runs through the simple LLM tutorial in langchain docs.
"""

import argparse
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_API_KEY'] = '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'


def main(to_language: str, user_message: str) -> None:
    """
    The main method where execution starts.
    """
    base_url: str = 'http://localhost:1234/v1'
    api_key: str = 'lm_studio'

    # make the prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', 'Translate the following into {language}:'),
        ('user', '{text}'),
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
        'language': to_language,
        'text': user_message,
    }

    result = chain.invoke(data)
    print(result)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--to_lang', required=True, type=str,
    )
    arg_parser.add_argument(
        '--user_message', required=True, type=str,
    )
    args = arg_parser.parse_args()

    main(
        args.to_lang, args.user_message,
    )
