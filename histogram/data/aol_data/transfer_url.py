import math
import re


def extract_and_calculate_b_value(file_path, output_file, N, b):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    url_pattern = r'http[s]?://(?:www\.)?([a-zA-Z0-9]+)'
    prefixes = []

    for line in lines:
        match = re.search(url_pattern, line)
        if match:
            domain = match.group(1)
            prefix = domain[:3]
            prefixes.append(prefix)
            if len(prefixes) >= N:
                break

    with open(output_file, 'w') as out_file:
        out_file.write(f"{N}\n")
        out_file.write(f"{2 ** b}\n")
        for prefix in prefixes:
            binary_prefix = ''.join(format(ord(char), '08b') for char in prefix)

            if len(binary_prefix) >= b:
                binary_suffix = binary_prefix[-b:]
                decimal_value = int(binary_suffix, 2)
            else:
                decimal_value = int(binary_prefix, 2)

            out_file.write(str(decimal_value) + '\n')

file_path = r'user-ct-test-collection-01.txt'


for B in {131072}:
    for n in {131072}:
        b = int(math.log2(B))
        output_file = f"trans_01_n_{n}_b_{b}_B_{B}.txt"

        extract_and_calculate_b_value(file_path, output_file, n, b)
        print(f"data is saved to {output_file}")
