"""
This script runs through the simple LLM tutorial in langchain docs.
"""

from langchain_openai import ChatOpenAI
import os
from langchain_core.messages import HumanMessage, SystemMessage


os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_API_KEY'] = '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'

def main():
    """
    The main method where execution starts.
    """
    base_url: str = 'http://localhost:1234/v1'
    api_key: str = 'lm_studio'

    model = ChatOpenAI(
        temperature=0.0, base_url=base_url, api_key=api_key,
    )
    
    messages = [
        SystemMessage(content='Translate the following from English to Italian.'),
        HumanMessage(content='Hi!'),
    ]
    output = model.invoke(messages)
    content = output.content
    print(content)


if __name__ == '__main__':
    main()
