"""
This script implements a chatbot by following the Langchain tutorial.
"""

from typing import Sequence

from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                     trim_messages)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.base import RunnableLambda
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, add_messages
from typing_extensions import Annotated, TypedDict


class State(TypedDict):
    """
    The state of the chatbot.
    """
    chat_messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


class ChatApp:
    """
    A simple chatbot application using a language model.
    """
    model: ChatOpenAI
    workflow: StateGraph
    memory: MemorySaver
    trimmer: RunnableLambda
    app: any

    def _init_model(self):
        # make the model
        base_url: str = 'http://localhost:1234/v1'
        api_key: str = 'lm_studio'

        self.model = ChatOpenAI(temperature=0.0, base_url=base_url, api_key=api_key)

    def call_model(self, state: State) -> dict:
        """
        Calls the language model.
        """
        prompt = ChatPromptTemplate.from_messages([
            ('system', 'Answer the questions in {language}'),
            MessagesPlaceholder(variable_name='chat_messages'),
        ])
        chain = prompt | self.model

        # trim the messages
        state['chat_messages'] = self.trimmer.invoke(state['chat_messages'])

        response = chain.invoke(state)
        return {
            'chat_messages': response,
        }

    def __init__(self):
        # make the model
        self._init_model()
        # make memory
        self.memory = MemorySaver()
        # trimmer to manage context length
        self.trimmer = trim_messages(  # pylint: disable=no-value-for-parameter
            max_tokens=1000, strategy='last', token_counter=self.model, include_system=True,
            allow_partial=False, start_on='human',
        )

        # define the workflow
        self.workflow = StateGraph(state_schema=State)
        self.workflow.add_edge(START, 'model')
        self.workflow.add_node('model', self.call_model)

        self.app = self.workflow.compile(checkpointer=self.memory)

    def run(self) -> None:
        """
        This runs the chatbot.
        """
        config: str = {'configurable': {'thread_id': '123456'}}

        language: str = input('Enter the language in which you want to chat: ')
        language = language.strip()

        print('Enter your message (enter STOP to end) at the prompt.\n')
        while True:
            user_message: str = input('Your message: ')
            user_message = user_message.strip()

            if user_message.lower() == 'stop':
                break

            input_messages = [HumanMessage(user_message)]
            state = State(chat_messages=input_messages, language=language)

            # stream the messages
            print('AI Response:')
            for chunk, _ in self.app.stream(state, config, stream_mode='messages'):
                if isinstance(chunk, AIMessage):
                    print(chunk.content, end='', flush=True)
            print()


def main() -> None:
    """
    The main method to start execution.
    """
    chat_app = ChatApp()
    chat_app.run()


if __name__ == '__main__':
    main()
