"""
This script implements a chatbot by following the Langchain tutorial.
"""

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph


class ChatApp:
    """
    A simple chatbot application using a language model.
    """
    model: ChatOpenAI
    workflow: StateGraph
    memory: MemorySaver
    app: any

    def _init_model(self):
        # make the model
        base_url: str = 'http://localhost:1234/v1'
        api_key: str = 'lm_studio'

        self.model = ChatOpenAI(temperature=0.0, base_url=base_url, api_key=api_key)

    def call_model(self, state: MessagesState) -> dict:
        """
        Calls the language model.
        """
        prompt = ChatPromptTemplate.from_messages([
            ('system', 'You will talk like a pirate. Answer the questions to the best of your ability.'),
            MessagesPlaceholder(variable_name='chat_messages'),
        ])
        chain = prompt | self.model
        response = chain.invoke({
            'chat_messages': state['messages'],
        })
        return {
            'messages': response,
        }

    def __init__(self):
        # make the model
        self._init_model()
        # make memory
        self.memory = MemorySaver()

        # define the workflow
        self.workflow = StateGraph(state_schema=MessagesState)
        self.workflow.add_edge(START, 'model')
        self.workflow.add_node('model', self.call_model)

        self.app = self.workflow.compile(checkpointer=self.memory)

    def run(self) -> None:
        """
        This runs the chatbot.
        """
        config: str = {'configurable': {'thread_id': '123456'}}

        print('Enter your message (enter STOP to end) at the prompt.\n')
        while True:
            user_message: str = input('Your message: ')
            user_message = user_message.strip()

            if user_message.lower() == 'stop':
                break

            input_messages = [HumanMessage(user_message)]
            output = self.app.invoke(
                {'messages': input_messages}, config
            )
            print(f'AI Response: {output["messages"][-1].content}')


def main() -> None:
    """
    The main method to start execution.
    """
    chat_app = ChatApp()
    chat_app.run()


if __name__ == '__main__':
    main()
