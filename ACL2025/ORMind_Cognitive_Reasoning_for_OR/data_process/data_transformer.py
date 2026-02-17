import json
import os


def process_json(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as file:
        original_data = json.load(file)

    new_data = {
        "background": original_data.get("description", ""),
        "constraints": [
            constraint["description"] for constraint in original_data.get("model", {}).get("constraint", [])
        ],
        "objective": original_data.get("model", {}).get("objective", [{}])[0].get("description", ""),
        "description": original_data.get("description", ""),
        "parameters": []
    }

    for set_item in original_data.get("model", {}).get("set", []):
        new_data["parameters"].append({
            "symbol": set_item.get("name", ""),
            "definition": set_item.get("description", ""),
            "shape": []
        })

    for param in original_data.get("model", {}).get("parameter", []):
        shape = []
        domain = param.get("domain", "")
        if "," in domain:
            shape = domain.replace("{", "").replace("}", "").split(",")
            shape = [item.split("<in>")[1].strip() if "<in>" in item else item.strip() for item in shape]
        new_data["parameters"].append({
            "symbol": param.get("name", ""),
            "definition": param.get("description", ""),
            "shape": shape
        })

    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(new_data, file, indent=4, ensure_ascii=False)


def main():
    input_directory = 'dataset/ComplexOR/seeds_model'
    output_base_directory = 'dataset/ComplexOR_new/'


    existing_folders = set(os.listdir(output_base_directory))

    for filename in os.listdir(input_directory):
        if filename.endswith('.json'):
            base_name = os.path.splitext(filename)[0]
            folder_name = base_name.split('_')[0]

            if folder_name not in existing_folders:
                print(f"Skipping {filename} as folder {folder_name} does not exist.")
                continue

            new_folder_path = os.path.join(output_base_directory, folder_name)

            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(new_folder_path, "input_targets.json")

            try:
                process_json(input_path, output_path)
                print(f"Processed {filename} and saved to {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    print("All files processed.")


if __name__ == "__main__":
    main()
