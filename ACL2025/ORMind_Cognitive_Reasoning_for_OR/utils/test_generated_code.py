import importlib
import json
import math
import os
import sys

from utils.result import Result


class NullWriter:
    def write(self, s):
        pass


def test_generated_code(problem, samples, log_file=None):
    log_file = log_file or NullWriter()

    try:

        from temp import generated_code
        importlib.reload(generated_code)
    except BaseException as e:
        log_file.write('There is grammar error in generated code!\n')
        log_file.write(str(e) + '\n')
        return Result.COMPILE_ERROR

    try:
        if "prob" in problem:
            func = getattr(generated_code, problem)
        else:
            func = getattr(generated_code, "solve")
    except AttributeError as e:
        log_file.write('Cannot load function!\n')
        log_file.write(str(e) + '\n')
        return Result.COMPILE_ERROR

    post_process = None
    if os.path.exists(os.path.join('../dataset', problem, 'data_process.py')):
        data_process = importlib.import_module('dataset.' + problem + '.data_process')
        if hasattr(data_process, 'post_process'):
            post_process = data_process.post_process

    total_num = len(samples)
    passed_num = 0
    is_re = False
    for i, sample in enumerate(samples):
        try:
            if "prob" in problem:
                output = func(**sample['input'])
            else:
                output = func(sample['input'])
                if isinstance(output, dict):
                    if output["status"] == "Optimal":
                        output = output["objective_value"]
                    else:
                        output = output["status"]
        except BaseException as e:
            is_re = True
            log_file.write('=' * 15 + f'test sample {i}' + '=' * 15 + '\n')
            log_file.write('Runtime Error\n')
            log_file.write(str(e) + '\n\n')
            continue
        if post_process is not None:
            output = post_process(*output)

        if len(sample['output']) == 1:
            ground_truth = sample['output'][0]
        else:
            ground_truth = tuple(sample['output'])

        print('=' * 20)
        print(output)
        print(ground_truth)
        print()
        log_file.write('=' * 15 + f'test sample {i}' + '=' * 15 + '\n')
        log_file.write('Program Output:\n')
        log_file.write(str(output) + '\n\n')
        log_file.write('Ground Truth:\n')
        log_file.write(str(ground_truth) + '\n')
        if isinstance(output, tuple):
            output = output[0]
        if output is not None and ground_truth is not None:
            try:
                if type(output) in [int, float, complex]:
                    is_passed = math.isclose(output, ground_truth, rel_tol=1e-3, abs_tol=2e-1)
                else:
                    is_passed = output == ground_truth
            except BaseException as e:
                is_passed = False
        elif output is None and ground_truth is None:
            is_passed = True
        else:
            is_passed = False
        # is_passed = (output == ground_truth)
        if is_passed:
            passed_num += 1
        log_file.write(f'Is passed: {is_passed}\n')
        log_file.write('\n')
        # assert output == tuple(sample['output']), f'Test failed:\nprogram output: {output}\nground truth: {tuple(sample["output"])}'
    # print('Test passed!!!')
    log_file.write('\n\n')
    log_file.write(f'{passed_num}/{total_num} passed\n')
    is_correct = (passed_num == total_num)
    log_file.write(f'is correct: {is_correct}\n')

    if is_re:
        return Result.RUNTIME_ERROR
    if is_correct:
        return Result.ACCEPT
    else:
        return Result.WRONG_ANSWER


def run_generated_code(problem, samples):
    try:
        from temp import generated_code
        importlib.reload(sys.modules['temp.generated_code'])
    except BaseException as e:
        error_message = f"The previous code has compile error: {str(e)}, you need to fix it."
        return error_message

    try:
        if "prob" in problem:
            func = getattr(generated_code, problem)
        else:
            func = getattr(generated_code, "solve")
    except AttributeError as e:
        error_message = f"The previous code has compile error: {str(e)}, you need to fix it."
        return error_message

    results = []
    for i, sample in enumerate(samples):
        try:
            if "prob_" in problem:
                output = func(**sample['input'])
            else:
                output = func(sample['input'])
            results.append({"sample": i, "status": "success", "output": output})
        except BaseException as e:
            error_message = f"The previous code has running-time error: {str(e)}, you need to fix it."
            # print(error_message)  # 打印错误信息
            # results.append({"sample": i, "status": "error", "error_message": error_message})
            return error_message

        return output


# def run_eval_code(samples):
#     try:
#         import eval_code
#         importlib.reload(sys.modules['eval_code'])
#
#     except BaseException as e:
#         return Result.COMPILE_ERROR
#
#     try:
#         func = getattr(eval_code, "check_constraints")
#     except AttributeError as e:
#
#         return Result.COMPILE_ERROR
#
#     for i, sample in enumerate(samples):
#         # output = func(**sample)
#         try:
#             output = func(**sample)
#         except BaseException as e:
#             return e
#         return output


def run_eval_code(samples):
    from temp import eval_code
    importlib.reload(sys.modules['temp.eval_code'])

    func = getattr(eval_code, "counterfactual_solution_analysis")

    for i, sample in enumerate(samples):
        output = func(**sample)

        return output


def read_test_samples(dataset, problem):
    with open(os.path.join('dataset', dataset, problem, 'data.json'), 'r', encoding='utf8') as f:
        test_samples = json.load(f)
    return test_samples


def test_origin_output(output, samples, log_file=None):
    log_file = log_file or NullWriter()

    total_num = len(samples)
    passed_num = 0
    for i, sample in enumerate(samples):

        if len(sample['output']) == 1:
            ground_truth = sample['output'][0]
        else:
            ground_truth = tuple(sample['output'])

        print('=' * 20)
        print(output)
        print(ground_truth)
        print()
        log_file.write('=' * 15 + f'test sample {i}' + '=' * 15 + '\n')
        log_file.write('Program Output:\n')
        log_file.write(str(output) + '\n\n')
        log_file.write('Ground Truth:\n')
        log_file.write(str(ground_truth) + '\n')
        if isinstance(output, tuple):
            output = output[0]
        try:
            is_passed = math.isclose(output, ground_truth, rel_tol=1e-3, abs_tol=2e-1)
        except BaseException as e:
            is_passed = False
        # is_passed = (output == ground_truth)
        if is_passed:
            passed_num += 1
        log_file.write(f'Is passed: {is_passed}\n')
        log_file.write('\n')
        # assert output == tuple(sample['output']), f'Test failed:\nprogram output: {output}\nground truth: {tuple(sample["output"])}'
    # print('Test passed!!!')
    log_file.write('\n\n')
    log_file.write(f'{passed_num}/{total_num} passed\n')
    is_correct = (passed_num == total_num)
    log_file.write(f'is correct: {is_correct}\n')

    if is_correct:
        return Result.ACCEPT
    else:
        return Result.WRONG_ANSWER


if __name__ == '__main__':
    dataset = 'LPWP'
    problem = 'prob_245'
    test_samples = read_test_samples(dataset, problem)
    test_generated_code(problem, test_samples)
