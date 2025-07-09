import math
import re


def extract_and_calculate_b_value(file_path, output_file, N, b):
    # 读取文件
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 定义正则表达式来提取 URL
    url_pattern = r'http[s]?://(?:www\.)?([a-zA-Z0-9]+)'

    # 列表保存处理结果
    prefixes = []

    # 处理每一行
    for line in lines:
        match = re.search(url_pattern, line)
        if match:
            # 获取匹配的 URL 中的域名部分
            domain = match.group(1)

            # 提取域名的前三个字母
            prefix = domain[:3]

            # 保存提取的三个字母
            prefixes.append(prefix)

            # 如果已经提取到N个URL，停止提取
            if len(prefixes) >= N:
                break


    # 将结果写入输出文件
    with open(output_file, 'w') as out_file:
        # 第一行写入提取的 URL 个数 N 和 B
        out_file.write(f"{N}\n")
        out_file.write(f"{2 ** b}\n")
        # 从第二行开始写入每个 URL 提取的前三个字母,并转换二进制形式
        for prefix in prefixes:
            # 将每个字母转换为二进制字符串，并拼接成一个二进制字符串
            binary_prefix = ''.join(format(ord(char), '08b') for char in prefix)

            # 如果二进制字符串的长度不足b位，则直接使用二进制值
            if len(binary_prefix) >= b:
                # 获取二进制数的后b位
                binary_suffix = binary_prefix[-b:]
                # 将这后b位二进制数转换成十进制数
                decimal_value = int(binary_suffix, 2)
            else:
                # 如果不足b位，直接将整个二进制字符串转换为十进制
                decimal_value = int(binary_prefix, 2)

            out_file.write(str(decimal_value) + '\n')


# 设置文件路径和输出文件路径
file_path = r'user-ct-test-collection-01.txt'


# 自定义的 N 和 b
for B in {131072}:
    for n in {131072}:
        b = int(math.log2(B))  # 你希望取二进制数的后 b 位进行转换
        output_file = f"trans_01_n_{n}_b_{b}_B_{B}.txt"

        # 执行函数
        extract_and_calculate_b_value(file_path, output_file, n, b)
        print(f"处理后的数据已保存到 {output_file}")