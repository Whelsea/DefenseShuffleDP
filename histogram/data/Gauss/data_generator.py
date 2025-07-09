import numpy as np
import random
import os


def generate_gaussian_data(n, B, mu, sigma, output_path):
    """
    生成高斯分布数据，先对 [1, B] 随机排列，再用高斯分布索引采样
    并清洗结果：避免BOM、空行、非数字行

    参数:
        n: 数据点数量
        B: 数据范围 [1, B]
        mu: 高斯分布均值 (建议设为 B/2 附近)
        sigma: 高斯分布标准差 (控制数据集中程度)
        output_path: 输出文件路径
    """
    # Step 1: 随机打乱 [1, B] 作为值域
    domain = list(range(1, B))
    random.shuffle(domain)

    # Step 2: 生成高斯分布的概率密度
    # 生成B个点的高斯分布概率，并归一化
    x = np.arange(1, B)
    weights = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    prob = weights / weights.sum()

    # Step 3: 按概率采样
    indices = np.random.choice(B - 1, size=n, replace=True, p=prob)
    samples = [domain[i] for i in indices]

    # Step 4: 写入临时文件（防止中间中断）
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
    print(f" Generated {len(cleaned)} valid entries with Gaussian(mu={mu}, sigma={sigma})")
    print(f" Saved to: {os.path.abspath(output_path)}")


# 示例用法
if __name__ == "__main__":
    for B in {131072}:  # 数据范围
        for n in {131072}:  # 数据点数量
            mu = B // 5
            sigma = B / 5
            output_path = f"./Gauss_n{n}B{B}"

            generate_gaussian_data(n, B, mu, sigma, output_path)
