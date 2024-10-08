"""
This agent adds numbers.
"""
from langchain.agents import AgentType, initialize_agent
from llms.gpt import GptLlm
from tools.add_numbers import add_tool
from transformers import AutoModelForCausalLM, AutoTokenizer


class AddNumbersAgent:
    """
    The agent to add numbers.
    """

    def __init__(self) -> None:
        super().__init__()

        self.model_name: str = 'gpt2'

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        self.llm = GptLlm(model=self.model, tokenizer=self.tokenizer)

        self.agent = initialize_agent(
            tools=[add_tool], agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, llm=self.llm,
        )

    def run(self, user_input):
        """
        Runs the agent.
        """
        return self.agent.invoke(user_input, handle_parsing_errors=True)
