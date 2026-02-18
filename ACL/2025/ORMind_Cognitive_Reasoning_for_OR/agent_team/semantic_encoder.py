from agent_team.base_expert import BaseExpert
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


class SemanticEncoder(BaseExpert):
    ROLE_DESCRIPTION = 'You are an assistant that extracts parameters and their types or shape from the given problem.'
    FORWARD_TASK = '''
    Please review the following example and extract the parameters along with their concise definitions:
    {problem_example}
    The comment from your colleague is:
    {comment_text}
    Your output should be in JSON format as follows:
    {{
        "Parameter1": {{"Type": "The parameter's data type or shape", "Definition": "A brief definition of the parameter"}},
        "Parameter2": {{"Type": "The parameter's data type or shape", "Definition": "A brief definition of the parameter"}},
        ...
    }}
    Provide only the requested JSON output without any additional information.
    '''

    def __init__(self, model,temperature=0,base_url=None,api_key=None):
        super().__init__(
            name='Parameter Extractor',
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


    def forward(self, problem,comment_pool=None):
        if isinstance(problem["description"],dict):
            problem_example= problem["description"]["parameters"]
        else:
            problem_example = problem["code_example"]
        output = self.forward_chain.invoke(
            {"problem_example": problem_example,
             "comment_text": comment_pool.get_current_comment_text()}
        ).content

        return output

