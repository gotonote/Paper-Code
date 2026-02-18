from agent_team.base_expert import BaseExpert
from utils.utils import extract_code_from_string

class Compiler(BaseExpert):
    ROLE_DESCRIPTION = 'You are a Python programmer specializing in operations research and optimization.'
    FORWARD_TASK = '''You are presented with a specific problem and tasked with developing an efficient Python program to solve it.
    The original problem is as follows:
    {problem_description}
    Your colleague has constructed a mathematical model for reference:
    {comments_text}
    Please note that this model may contain errors and is used as a reference. 
    You can analyze the problem step by step and provide your own code.
    Requirements:
    1. Use the PuLP library for implementation.
    2. Provide a function that solves the problem.
    3. Do not include code usage examples or specific variable values.
    4. Focus on creating a general, reusable solution.'''


    def __init__(self, model,temperature=0,base_url=None,api_key=None):
        super().__init__(
            name='Programming Expert',
            description='Skilled in programming and coding, capable of implementing the optimization solution in a programming language.',
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key
        )

    def forward(self, problem, comment_pool):
        if isinstance(problem["description"], dict):
            problem_description = ' '.join(problem["description"].values())
        else:
            problem_description = problem["description"]
        comments_text = comment_pool.get_current_comment_text()

        output = self.forward_chain.invoke({
            "problem_description":problem_description,
            "comments_text":comments_text}
        ).content
        code=extract_code_from_string(output)
        return code

