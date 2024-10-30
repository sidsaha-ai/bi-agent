"""
This scripts implements a chatbot that uses an LLM to gauge the sentiment of the user's message and
repsonsds accordinly.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph

from typing_extensions import TypedDict, NotRequired

class SentimentInput(TypedDict):
    system_prompt: str
    user_prompt: str


class WorkflowState(TypedDict):
    user_message: str
    sentiment: NotRequired[str]


class SentimentalChatApp:
    """
    A sentinmental chatbot application using a language model.
    """
    model: ChatOpenAI
    workflow: StateGraph
    app: any

    def _init_model(self):
        # make the model
        base_url: str = 'http://localhost:1234/v1'
        api_key: str = 'lm_studio'

        self.model = ChatOpenAI(temperature=0.0, base_url=base_url, api_key=api_key)
    
    def __init__(self) -> None:
        # make the model
        self._init_model()

        # make the workflow
        self.workflow = StateGraph(state_schema=WorkflowState)
        self.workflow.add_edge(START, 'find_sentiment')
        self.workflow.add_node('find_sentiment', self.find_sentiment)

        # make the app
        self.app = self.workflow.compile()
    
    def find_sentiment(self, state: WorkflowState) -> str:
        """
        Use the model to find the sentiment of the user's message.
        """
        system_prompt: str = """
        You will be given a message and you have to gauge the sentiment of the message. If the sentiment is
        positive, output "positive", if the sentiment is negative, output "negative", and if the sentiment is
        neutral, output "neutral". Below are a few examples - 

        # Example 1
        Message: The day is so bright and sunny. I feel it very invigorating.
        AI: positive

        # Example 2
        Message: My parents just informed me that my dog died.
        AI: negative

        # Example 3
        Message: My name is Sid, and I live in Bangalore.
        AI: neutral.

        Important Instruction: Only respond with the string "positive", "negative", or "neutral". There should be
        nothing else in your response. Your response should just be one word.
        """
        user_prompt: str = f'''
        Below is the message you have to work on. Find the sentiment of the message to the best of your ability.
        {state['user_message']}
        '''
        prompt = ChatPromptTemplate.from_messages([
            ('system', '{system_prompt}'),
            ('user', '{user_prompt}'),
        ])

        chain = prompt | self.model

        sentinment_input = SentimentInput(system_prompt=system_prompt, user_prompt=user_prompt)
        response = chain.invoke(sentinment_input)

        print(response.content)
        state['sentiment'] = response.content
    
    def run(self) -> None:
        """
        Method to run the chat app.
        """
        user_message: str = input('Tell me what is going on today?: ')
        user_message = user_message.strip()

        state: WorkflowState = WorkflowState(user_message=user_message)
        response = self.app.invoke(state)
        print(response)


if __name__ == '__main__':
    app = SentimentalChatApp()
    app.run()
