import os

def clean_csv_remove_commas(input_path, output_path, expected_fields=10):
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')

            # 如果列数比预期多，说明某些字段中多了逗号
            if len(parts) > expected_fields:
                # 修复思路：保留第1列为ID，最后(expected_fields - 2)列为正规字段
                id_part = parts[0]
                tail_fields = parts[-(expected_fields - 2):]

                # 中间部分合并为职位字段，去掉内部逗号
                position_field = ' '.join(parts[1:len(parts) - (expected_fields - 2)]).replace(',', ' ').strip()

                fixed_line = [id_part, position_field] + tail_fields

            elif len(parts) == expected_fields:
                fixed_line = parts

            else:
                print(f"[Warning] Line {lineno} has only {len(parts)} fields. Skipped.")
                continue

            fout.write(','.join(fixed_line) + '\n')

    print(f"✅ Cleaned CSV written to: {output_path}")


# 示例使用
if __name__ == '__main__':
    input_file = './data_raw.csv'
    output_file = './data.csv'

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    clean_csv_remove_commas(input_file, output_file)
