"""
This agent adds numbers.
"""
from langchain.agents import AgentType, initialize_agent
from llms.lm_studio import LMStudioLLM
from tools.add_numbers import add_tool


class AddNumbersAgent:
    """
    The agent to add numbers.
    """

    def __init__(self) -> None:
        super().__init__()

        self.model_id: str = 'llama-3.2-3b-instruct-4bit'
        self.lm_url: str = 'http://localhost:1234/v1'

        self.llm = LMStudioLLM(lm_url=self.lm_url, model_id=self.model_id)

        self.agent = initialize_agent(
            tools=[add_tool], agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, llm=self.llm,
        )

    def run(self, user_input):
        """
        Runs the agent.
        """
        prompt = self._generate_prompt(user_input)
        return self.agent.invoke(prompt, handle_parsing_errors=True)

    def _generate_prompt(self, user_input: str) -> str:
        """
        Generates and returns the prompt.
        """
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
        Question: {user_input}
        '''
        return prompt
