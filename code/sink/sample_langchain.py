from langchain import LLMChain, PromptTemplate
from langchain.agents import Tool, initialize_agent
from langchain.chains import LLMChain
from llms.lm_studio import LMStudioLLM


def add_numbers(numbers_str: str) -> int:
    try:
        numbers = [int(num.strip()) for num in numbers_str.split(',')]
        return sum(numbers)
    except ValueError:
        return 'Invalid numbers'


def main():
    model_id: str = 'llama-3.2-3b-instruct-4bit'
    lm_url: str = 'http://localhost:1234/v1'
    llm = LMStudioLLM(lm_url=lm_url, model_id=model_id)

    prompt_template = PromptTemplate(
        input_variables=['user_question'],
        template='''
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

        Below is the question you have to work on -
        Question: {user_question}
        '''
    )

    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    
    user_query = 'I need to sum the numbers 10, 12, and 14.'
    numbers = llm_chain.run({'user_question': user_query})
    print(f'==== NUMBERS: {numbers}')

    add_tool = Tool(
        name='AddTool', func=add_numbers, description='Sums a list of numbers provided as a comma-separated string.',
    )
    
    tools = [add_tool]
    agent = initialize_agent(
        tools=tools, agent='zero-shot-react-description', verbose=True,
    )

    user_query = 'I need to sum the numbers 10, 12, and 14.'

    response = agent.run(numbers)
    print(response)

if __name__ == '__main__':
    main()
