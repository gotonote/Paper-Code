
from agent_team.base_expert import BaseExpert


class Reasoner(BaseExpert):
    ROLE_DESCRIPTION = 'You are a counterfactual thinker for optimization problem results.'

    FORWARD_TASK = '''Analyze the following optimization problem:
    {problem_description}

    Task: Write a Python function that identifies which specific constraints or conditions in the given problem 
    are not satisfied. This condition will need modification to achieve a valid and optimal solution.

    Function specifications:
    - Input arguments and their types: {input_content}
    - Adhere to the given data types.
    - Reference this code structure: {code_example}
    - Import the necessary libraries.

    Notes:
    The code example is only for reference in terms of format and structure. Generate code specifically for the given problem, not based on any examples. 
    All specific constraints should be determined based on the problem description provided. 
    Make sure to include checks for all constraints mentioned in the problem description. Don't give any Example usages.
    '''

    BACKWARD_TASK = '''Analyze the feedback from your previous code:

    Previous code:
    {previous_code}

    Error message:
    {feedback}

    Please identify:
    1. The cause of this error.
    2. The specific problematic code section.
    
    Prior Knowledge for reference:
    1. The pulp not support 'LpVariable' divide 'int', you need to use multiply to replace it.
    2. '>' are not supported between instances of 'LpAffineExpression' and 'int', you need to use ">=".
    3. name 'lp...' is not defined, you may use "from pulp import *" to include all.
    Your answer must be concise. A one-sentence answer is sufficient.
    '''

    def __init__(self, model,temperature=0,base_url=None,api_key=None):
        super().__init__(
            name='Causal specialist',
            description='An special expert that generates the test data and test correctness.',
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key
        )

    def forward(self, problem,code_example,input_content="Not applicable in this problem"):
        answer = self.forward_chain.invoke({
            "problem_description": problem['description'],
            "code_example": code_example,
            "input_content": input_content
        }).content

        return answer
    
    def backward(self, previous_code,feedback_pool):
        output = self.backward_chain.invoke({
            "previous_code": previous_code,
            "feedback": feedback_pool.get_closet_comment_text()
        }).content
        return output






