import os
import re
import statistics


def count_token(folder_path):
    pattern = r'.*_test_log\.txt$'
    prompt_tokens = []
    total_files = 0

    for filename in os.listdir(folder_path):
        if re.match(pattern, filename):
            total_files += 1
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                match = re.search(r'Prompt Tokens: (\d+)', content)
                if match:
                    tokens = int(match.group(1))
                    prompt_tokens.append(tokens)

    if prompt_tokens:
        avg_tokens = statistics.mean(prompt_tokens)
        std_dev = statistics.stdev(prompt_tokens)

        return {
            'average': avg_tokens,
            'std_dev': std_dev,
            'total_files': total_files
        }
    else:
        return None


folder_path = '../../log/run_LPWP_3.5_new'
result = count_token(folder_path)

if result:
    print(f"Average Prompt Tokens: {result['average']:.2f}")
    print(f"Standard Deviation: ±{result['std_dev']:.2f}")
    print(f"Total files processed: {result['total_files']}")
else:
    print("No matching files found or no token information available.")
