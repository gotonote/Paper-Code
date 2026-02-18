import json
import os

import numpy as np

from agent_team import (
    TerminologyInterpreter,
    code_reviewer,
    ModelingExpert,
    Compiler,
    SemanticEncoder,
    Conductor
)
from agent_team.supervisor import Supervisor
from agent_team.reasoner import Reasoner
from utils.comment import Comment
from utils.comment_pool import CommentPool
from utils.test_generated_code import run_generated_code, run_eval_code
from utils.utils import extract_code_from_string, format_constraint_results, get_dict_values_as_string


def causal_agent(problem_name,
                 problem,
                 model_name,
                 ifCausal,
                 ifSyntax,
                 ifCon,
                 temperature,
                 attention,
                 path,
                 test_samples,
                 custom_base_url=None,
                 api_key=None):
    """Run Chain of Experts pipeline
    
    Args:
        problem: a dict of problem_description and code_example.
    
    Return:
        code: code of problem
    """
    all_experts = [
        # TerminologyInterpreter(model_name,temperature,custom_base_url,api_key),
        SemanticEncoder(model_name, temperature, custom_base_url, api_key),
        ModelingExpert(model_name, temperature, custom_base_url, api_key),
        Compiler(model_name, temperature, custom_base_url, api_key),
        # CodeReviewer(model_name,temperature,custom_base_url,api_key),
    ]
    num_experts = len(all_experts)
    director = Supervisor(model_name, temperature, base_url=custom_base_url, api_key=api_key)
    comment_pool = CommentPool(all_experts, visible_matrix=np.ones((num_experts, num_experts)))
    conductor = Conductor(model_name, temperature, base_url=custom_base_url)
    expert_stack = []

    for i in range(len(all_experts)):
        if not ifCon:
            next_expert = all_experts[i]
        else:
            next_expert = conductor.forward(problem, comment_pool, max_collaborate_nums=len(all_experts))
        print(f'Choose next expert: {next_expert.name}')
        comment_text = next_expert.forward(problem, comment_pool)
        print(f'Given comment:\n{comment_text}')
        comment_pool.add_comment(Comment(next_expert, comment_text))
        expert_stack.append(next_expert)

    original_answer = director.forward(problem, comment_pool, attention)
    answer = original_answer
    code = extract_code_from_string(original_answer)
    with open('temp/generated_code.py', 'w') as f:
        f.write(code)

    output = run_generated_code(problem_name, test_samples)

    if ifCausal:
        evaluator = Reasoner(model_name, base_url=custom_base_url, api_key=api_key)
        if ifSyntax:
            if isinstance(output, str):
                with open(os.path.join(path, f'{problem_name}_error_code.py'), 'w', encoding='utf8') as f:
                    f.write(code)
                comment_pool.add_comment(Comment(director, output))
                fail_reason = evaluator.backward(code, comment_pool)
                with open(os.path.join(path, f'{problem_name}_eval_result.txt'), 'w', encoding='utf8') as f:
                    f.write(str(fail_reason))
                comment_pool.add_comment(Comment(evaluator, str(fail_reason)))
                answer = director.backward(problem, comment_pool)
                code = extract_code_from_string(answer)
                with open('temp/generated_code.py', 'w') as f:
                    f.write(code)

        with open(os.path.join(path, f'{problem_name}_origin_code.py'), 'w', encoding='utf8') as f:
            f.write(code)
        # LPWP problem
        if "prob_" in problem_name:
            if isinstance(output, tuple) and len(output) == 3:
                obj, var1, var2 = output
                eval_samples = [{
                    "obj": obj,
                    "var1": var1,
                    "var2": var2
                }]

                with open(os.path.join('example/eval_code_example.py'), 'r', encoding='utf8') as f:
                    eval_code_example = f.read()
                input_content = get_dict_values_as_string(eval_samples[0])

                eval_answer = evaluator.forward(problem, eval_code_example, input_content)
                eval_code = extract_code_from_string(eval_answer)

                with open('temp/eval_code.py', 'w') as f:
                    f.write(eval_code)
                with open(os.path.join(path, f'{problem_name}_eval_code.py'), 'w', encoding='utf8') as f:
                    f.write(eval_code)

                try:
                    result = run_eval_code(eval_samples)
                except Exception as e:
                    with open(os.path.join(path, f'{problem_name}_eval_result.txt'), 'w', encoding='utf8') as f:
                        f.write(str(e))
                    result = None

                if isinstance(result, dict):
                    try:
                        check_result = format_constraint_results(result)
                    except:
                        check_result = "Don't need to modify any constraints!"

                    with open(os.path.join(path, f'{problem_name}_eval_result.txt'), 'w', encoding='utf8') as f:
                        f.write(check_result)

                    if check_result != "Don't need to modify any constraints!":
                        comment_pool.add_comment(Comment(evaluator, check_result))
                        answer = director.backward(problem, comment_pool)

            # ComplexOR problem
        elif "prob_" not in problem_name:
            if isinstance(output, dict) and "optimized_vars" in output.keys():
                input_arg = output["optimized_vars"]
                with open(os.path.join('dataset', "ComplexOR", problem_name, 'data.json'), 'r', encoding='utf8') as f:
                    input_arg["data"] = json.load(f)[0]["input"]

                eval_samples = [input_arg]

                with open(os.path.join('example/OR_eval_code_example.py'), 'r', encoding='utf8') as f:
                    eval_code_example = f.read()
                input_content = get_dict_values_as_string(eval_samples[0])
                eval_answer = evaluator.forward(problem, eval_code_example, input_content)
                eval_code = extract_code_from_string(eval_answer)
                with open('temp/eval_code.py', 'w') as f:
                    f.write(eval_code)
                with open(os.path.join(path, f'{problem_name}_eval_code.py'), 'w', encoding='utf8') as f:
                    f.write(eval_code)

                try:
                    result = run_eval_code(eval_samples)
                except Exception as e:
                    with open(os.path.join(path, f'{problem_name}_eval_result.txt'), 'w', encoding='utf8') as f:
                        f.write(str(e))
                    result = None

                if isinstance(result, dict):
                    try:
                        check_result = format_constraint_results(result)
                    except:
                        check_result = "Don't need to modify any constraints!"
                    with open(os.path.join(path, f'{problem_name}_eval_result.txt'), 'w', encoding='utf8') as f:
                        f.write(check_result)
                    if check_result != "Don't need to modify any constraints!":
                        comment_pool.add_comment(Comment(evaluator, check_result))
                        answer = director.backward(problem, comment_pool)

    if output is None:
        comment_pool.add_comment(Comment(evaluator, "the answer has no return"))
        answer = director.backward(problem, comment_pool)

    return answer, output
