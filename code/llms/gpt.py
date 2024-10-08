"""
This file defines an LLM wrapper to use with Llama.
"""

from langchain.llms.base import LLM
import torch
from pydantic import SkipValidation

class GptLlm(LLM):
    """
    The GPT LLM wrapper.
    """

    model: SkipValidation[any]
    tokenizer: SkipValidation[any]

    def _call(self, prompt: str, stop=None) -> str:
        """
        Calls the GPT LLM.
        """
        inputs = self.tokenizer(prompt, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=200)
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @property
    def _identifying_params(self) -> dict:
        """
        Returns some parameters to identify this LLM.
        """
        return {
            'model': 'GPT-2',
        }
    
    @property
    def _llm_type(self) -> str:
        """
        Returns the LLM Type.
        """
        return 'GPT'
