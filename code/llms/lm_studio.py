"""
Contains the abstract class for an LLM in the LM Studio.
"""
from langchain_core.language_models.llms import LLM
from openai import OpenAI


class LMStudioLLM(LLM):
    """
    Abstract class to represent an LLM in the LM Studio.
    """

    lm_url: str
    model_id: str

    def _call(self, prompt: str, **kwargs) -> str:  # pylint: disable=arguments-differ
        """
        Calls the LLM.
        """
        client = OpenAI(base_url=self.lm_url, api_key='lm_studio')

        completion = client.chat.completions.create(
            model=self.model_id,
            messages=[
                {'role': 'user', 'content': prompt},
            ],
        )
        if completion is None:
            return None
        if completion.choices is None or len(completion.choices) == 0:
            return None

        choice = completion.choices[0]
        if choice.message is None or choice.message.content is None:
            return None

        return choice.message.content

    @property
    def _identifying_params(self) -> dict:
        """
        Returns some parameters to identify this LLM.
        """
        return {
            'model_id': self.model_id,
        }

    @property
    def _llm_type(self) -> str:
        """
        Returns the LLM type.
        """
        return self.model_id
