import argparse
import os
import re
import time
from pathlib import Path

from langchain_community.callbacks import get_openai_callback
from tqdm import tqdm

from main import causal_agent
from utils.result import Result
from utils.test_generated_code import test_generated_code, read_test_samples, test_origin_output
from utils.utils import extract_code_from_string, read_problem, extract_number


def main():
    parser = argparse.ArgumentParser(description='Generate and test code.')
    parser.add_argument('--dataset', type=str, default='LPWP', help='Dataset name, "LPWP" or "ComplexOR"')
    parser.add_argument('--problem', type=str, default='prob_.*', help='Problem name')
    parser.add_argument('--log_dir', type=str, default='../log', help='The directory of log')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Base large language model')
    parser.add_argument('--ifCausal', type=bool, default=True, help='if use causal model')
    parser.add_argument('--ifSyntax', type=bool, default=True, help='if analyze syntax')
    parser.add_argument('--ifCon', type=bool, default=False, help='if use conductor')
    parser.add_argument('--temperature', type=float, default=0, help='temperature for llm')
    parser.add_argument('--attention', type=str, default='''Note: While certain parameters in the example may not be utilized, it is imperative to include all of them in the function definition. 
    The function must return a tuple, with the first element being the objective value. A dictionary is not permitted as the return type.''')
    parser.add_argument('--base_url', type=str, default="", help='openai base url')
    parser.add_argument('--api_key', type=str, default="", help='openai key')

    args = parser.parse_args()

    mached_problems = []
    for p in os.listdir(os.path.join('dataset', args.dataset)):
        if re.match(args.problem, p):
            mached_problems.append(p)

    mached_problems.sort(key=extract_number)
    total_num = len(mached_problems)
    if total_num == 0:
        print('No problem matched! Please check arguments.')
        exit(0)

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    log_dir_name = f'run_{args.dataset}_{str(round(time.time()))}'
    path = os.path.join(args.log_dir, log_dir_name)
    print(f'Save log to {path}')
    Path(path).mkdir(parents=True, exist_ok=True)

    correct_num = 0
    ce_num = 0
    re_num = 0
    pbar = tqdm(total=len(mached_problems))
    current_num = 0
    for problem in mached_problems:
        problem_data = read_problem(args.dataset, problem)
        test_samples = read_test_samples(args.dataset, problem)
        with get_openai_callback() as cb:
            answer, output = causal_agent(
                problem,
                problem_data,
                args.model,
                args.ifCausal,
                args.ifSyntax,
                args.ifCon,
                args.temperature,
                args.attention,
                path=path,
                test_samples=test_samples,
                custom_base_url=args.base_url,
                api_key=args.api_key)
            time.sleep(1)

        with open(os.path.join(path, f'{problem}_answer.txt'), 'w', encoding='utf8') as f:
            f.write(answer)

        code = extract_code_from_string(answer)

        with open(os.path.join(path, f'{problem}_generated_code.py'), 'w', encoding='utf8') as f:
            f.write(code)

        with open('temp/generated_code.py', 'w') as f:
            f.write(code)

        with open(os.path.join(path, f'{problem}_test_log.txt'), 'w', encoding='utf8') as f:
            result = test_generated_code(problem, test_samples, f)

        if result == Result.COMPILE_ERROR or result == Result.RUNTIME_ERROR:
            if isinstance(output, tuple) and len(output) == 3:
                with open(os.path.join(path, f'{problem}_test_log.txt'), 'w', encoding='utf8') as f:
                    result = test_origin_output(output[0], test_samples, f)

        with open(os.path.join(path, f'{problem}_test_log.txt'), 'a', encoding='utf8') as f:
            f.write(f"\n\nToken usage for problem {problem}:\n")
            f.write(str(cb))
            f.write("\n" + "-" * 50 + "\n")

        if result == Result.ACCEPT:
            correct_num += 1

        elif result == Result.COMPILE_ERROR:
            ce_num += 1
        elif result == Result.RUNTIME_ERROR:
            re_num += 1

        pbar.update()
        current_num += 1
        pbar.set_description(
            f'Accuracy: {correct_num / current_num * 100:.2f}% | Compile error: {ce_num / current_num * 100:.2f}% | Runtime error: {re_num / current_num * 100:.2f}%')

    print(f'Passed: {correct_num}/{total_num}')
    print(f'Accuracy: {correct_num / total_num * 100:.2f}%')
    print(f'Compile error: {ce_num / total_num * 100:.2f}%')
    print(f'Runtime error: {re_num / total_num * 100:.2f}%')


if __name__ == '__main__':
    main()
