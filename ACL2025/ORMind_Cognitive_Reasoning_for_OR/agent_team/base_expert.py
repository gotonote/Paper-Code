from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

class BaseExpert(object):

    def __init__(self, name, description, model,temperature=0, base_url=None,api_key=None):
        self.name = name
        self.description = description
        self.model = model

        self.llm = ChatOpenAI(
            model_name=model,
            temperature=temperature,
            openai_api_base=base_url,
            api_key=api_key,
            max_retries=0
        )
        self.forward_prompt_template = self.ROLE_DESCRIPTION + '\n' + self.FORWARD_TASK
        self.forward_prompt = PromptTemplate.from_template(self.forward_prompt_template)
        self.forward_chain = self.forward_prompt | self.llm

        if hasattr(self, 'BACKWARD_TASK'):
            self.backward_prompt_template = self.ROLE_DESCRIPTION + '\n' + self.BACKWARD_TASK
            self.backward_prompt = PromptTemplate.from_template(self.backward_prompt_template)
            self.backward_chain = self.backward_prompt | self.llm

    def forward(self, **kwargs):
        return self.forward_chain.invoke(kwargs).content

    def backward(self, **kwargs):
        if hasattr(self, 'backward_chain'):
            return self.backward_chain.invoke(kwargs).content
        else:
            raise NotImplementedError("Backward method not implemented for this expert.")

    def __str__(self):
        return f'{self.name}: {self.description}'