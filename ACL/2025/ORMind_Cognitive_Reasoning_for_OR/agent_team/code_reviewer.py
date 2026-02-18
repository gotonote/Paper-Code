from agent_team.base_expert import BaseExpert
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


class CodeReviewer(BaseExpert):

    ROLE_DESCRIPTION = 'You are a code reviewer that conducts thorough reviews of the implemented code to identify any errors, inefficiencies, or areas for improvement.'
    FORWARD_TASK = '''As a Code Reviewer, your responsibility is to conduct thorough reviews of implemented code related to optimization problems. 
You will identify possible errors, inefficiencies, or areas for improvement in the code, ensuring that it adheres to best practices and delivers optimal results. Now, here is the problem: 
{problem_description}. 

You are supposed to refer to the codes given by your colleagues from other aspects: {comments_text}'''


    def __init__(self, model,temperature,base_url,api_key=None):
        super().__init__(
            name='Code Reviewer',
            description='Skilled in programming and coding, capable of implementing the optimization solution in a programming language.',
            model=model   
        )
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


    def forward(self, problem, comment_pool):
        self.problem = problem
        comments_text = comment_pool.get_current_comment_text()
        output = self.forward_chain.invoke(
            {"problem_description": problem['description'],
             "comments_text": comments_text}
        ).content
        self.previous_code = output
        return output

