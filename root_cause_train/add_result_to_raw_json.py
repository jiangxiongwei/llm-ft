import json

def convert_json_with_result(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    json_arrays = []
    i = 0
    while i < len(lines):
        json_line = lines[i].strip()
        if json_line.startswith('['):
            try:
                json_array = json.loads(json_line)
                # Check if the next line is True or False
                if i + 1 < len(lines):
                    result_line = lines[i + 1].strip()
                    if result_line in ('True', 'False'):
                        json_array.append({"result": result_line})
                        json_arrays.append(json_array)
                        i += 2  # Skip the result line
                    else:
                        json_arrays.append(json_array)
                        i += 1
                else:
                    json_arrays.append(json_array)
                    i += 1
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at line {i + 1}: {e}")
                i += 1
        else:
            i += 1
    
    # Write to output file, each JSON array on a single line
    with open(output_file, 'w', encoding='utf-8') as f:
        for json_array in json_arrays:
            f.write(json.dumps(json_array, ensure_ascii=False) + '\n')
    print(f"Converted {len(json_arrays)} JSON arrays to {output_file}")

if __name__ == "__main__":
    input_file = "/home/linux/source/finetuning/root_cause_train/raw_data_from_daming.txt"
    output_file = "/home/linux/source/finetuning/root_cause_train/rc_data_with_result.txt"
    convert_json_with_result(input_file, output_file)