from agent_team.base_expert import BaseExpert


class ModelingExpert(BaseExpert):
    ROLE_DESCRIPTION = 'You are a modeling assistant specialized in the field of Operations Research.'

    FORWARD_TASK = '''Now the origin problem is as follows:
    {problem_description}
    You can use the parameters information from your colleague:
    {comments_text}
    The order of given parameters is random. You should clarify the meaning of each parameter to choose proper parameter to construct constraint.
    Give your Mathematical model of this problem.
    Your output format should be a JSON like this:
    {{
        "VARIABLES": "A concise description about variables and its shape or type",
        "CONSTRAINTS": "A mathematical Formula about constraints",
        "OBJECTIVE": "A mathematical Formula about objective"
    }}
    Don't give any other information.
    '''


    def __init__(self, model,temperature=0,base_url=None,api_key=None):
        super().__init__(
            name='Modeling Expert',
            description='Proficient in constructing mathematical optimization models based on the extracted information.',
            model=model  ,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key
        )

    def forward(self, problem, comment_pool):
        comments_text = comment_pool.get_closet_comment_text()
        if isinstance(problem["description"], dict):
            problem_description = "constraints: "+problem["description"]["constraints"]+" objective: "+problem["description"]["objective"]
        else:
            problem_description = problem["description"]

        output = self.forward_chain.invoke(
            {"problem_description": problem_description,
             "comments_text": comments_text}
        ).content

        self.previous_answer = output
        return output

