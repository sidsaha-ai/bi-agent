import argparse
from openai import OpenAI

def main(prompt):
    model_id = 'llama-3.2-3b-instruct'
    base_url = 'http://localhost:1234/v1'
    client = OpenAI(base_url=base_url, api_key='lm_studio')

    prompt = f'''
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
    Question: {prompt}
    '''
    print(prompt)

    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {'role': 'user', 'content': prompt},
        ],
    )
    print(f'Result: {completion.choices[0].message.content}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--prompt', type=str, required=True,
    )

    args = parser.parse_args()

    main(args.prompt)