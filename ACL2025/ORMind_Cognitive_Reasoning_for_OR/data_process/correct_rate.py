import os
import re


def count_cases(folder_path):
    total_files = 0
    correct_count = 0
    runtime_error_count = 0
    compile_error_count = 0
    pattern = r'.*_test_log\.txt$'

    for filename in os.listdir(folder_path):
        if re.match(pattern, filename):
            total_files += 1
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                if 'is correct: True' in content:
                    correct_count += 1
                elif 'Runtime Error' in content:
                    runtime_error_count += 1
                elif 'grammar error' in content or "Cannot load function!" in content:
                    compile_error_count += 1

    correct_ratio = correct_count / total_files if total_files > 0 else 0
    runtime_error_ratio = runtime_error_count / total_files if total_files > 0 else 0
    compile_error_ratio = compile_error_count / total_files if total_files > 0 else 0

    return total_files, correct_count, runtime_error_count, compile_error_count, correct_ratio, runtime_error_ratio, compile_error_ratio


folder_path = '../../log/'
total, correct, runtime_errors, compile_errors, correct_ratio, runtime_ratio, compile_ratio = count_cases(folder_path)

print(f"Total problem files: {total}")
print(f"Correct cases: {correct} ({correct_ratio:.2%})")
print(f"Runtime Error cases: {runtime_errors} ({runtime_ratio:.2%})")
print(f"Compile Error cases: {compile_errors} ({compile_ratio:.2%})")
print(
    f"Other cases: {total - correct - runtime_errors - compile_errors} ({1 - correct_ratio - runtime_ratio - compile_ratio:.2%})")
