from transformers import AutoTokenizer, AutoModelForCausalLM

import argparse

def main(prompt):
    model_name = 'meta-llama/Meta-Llama-3-8B'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompt = f'''
    Extract the numbers from the question. Just provide the numbers or say None. Below are a few examples -

    # Example 1
    Question: Tell me the sum of 10, 12, and 14.
    AI: 10, 12, 14.
    
    # Example 2
    Question: Can you get me the sum of 1 and 5.
    AI: 1, 5

    # Example 3
    Question: My name is Siddharth.
    AI: None

    Below is the question you have to work on -
    Question: {prompt}
    '''
    print(prompt)

    inputs = tokenizer(
        prompt, return_tensors='pt', padding=True, truncation=True,
    )
    
    tokens = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        max_length=len(prompt) + 50,
    )

    text = tokenizer.batch_decode(tokens)[0]
    print(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--prompt', type=str, required=True,
    )

    args = parser.parse_args()

    main(args.prompt)