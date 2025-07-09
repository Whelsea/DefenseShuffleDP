import numpy as np
import random
import os


def generate_zipf_data(n, B, alpha, output_path):
    """
    生成 Zipf 分布数据，先对 [1, B] 随机排列，再用 Zipf 分布索引采样
    并清洗结果：避免 BOM、空行、非数字行
    """
    # Step 1: 随机打乱 [1, B]
    domain = list(range(1, B))
    random.shuffle(domain)

    # Step 2: 构造 Zipf 分布的概率质量函数
    weights = np.array([1.0 / (i ** alpha) for i in range(1, B)])
    prob = weights / weights.sum()

    # Step 3: 从 [1, B] 中按概率采样 index，再用 domain[index] 得到真实值
    indices = np.random.choice(B - 1, size=n, replace=True, p=prob)
    samples = [domain[i] for i in indices]

    # Step 4: 写入临时文件（防止中间断）
    temp_path = output_path + ".tmp"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        f.write(f"{B}\n")
        for val in samples:
            f.write(f"{val}\n")

    # Step 5: 读取临时文件，清洗 + 重新写入
    cleaned = []
    with open(temp_path, "r", encoding="utf-8") as fin:
        for line in fin:
            stripped = line.strip()
            if stripped.isdigit():
                cleaned.append(stripped)

    with open(output_path, "w", encoding="utf-8") as fout:
        for val in cleaned:
            fout.write(val + "\n")

    os.remove(temp_path)
    print(f" Generated {len(cleaned)} valid entries with Zipf(alpha={alpha})")
    print(f" Saved to: {os.path.abspath(output_path)}")


# for d in {1048576, 16777216}:
#     query_id = random.randint(1, d + 1)
#     for n in {1024, 4096, 16384, 65536, 262144, 262144 * 4}:

# 示例用法
if __name__ == "__main__":
    for B in {2 ** 20}:
        for n in {2 ** 12, 2 ** 16, 2 ** 20, 2 ** 24}:
            alpha = 1.5  # Zipf 的 α 值（越大越偏）
            output_path = f"./Zip_n{n}B{B}"

            generate_zipf_data(n, B, alpha, output_path)
