import numpy as np
import random
import os


def generate_zipf_data(n, B, alpha, output_path):
    """
    Generate Zipf-distributed data by first shuffling [1, B],
    then sampling indices using a Zipf distribution.
    """
    # Step 1: Randomly shuffle [1, B)
    domain = list(range(1, B))
    random.shuffle(domain)

    # Step 2: Construct Zipf probability mass function
    weights = np.array([1.0 / (i ** alpha) for i in range(1, B)])
    prob = weights / weights.sum()

    # Step 3: Sample indices according to Zipf distribution, then map to shuffled domain
    indices = np.random.choice(B - 1, size=n, replace=True, p=prob)
    samples = [domain[i] for i in indices]

    # Step 4: Write to temporary file (to prevent data loss during interruption)
    temp_path = output_path + ".tmp"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        f.write(f"{B}\n")
        for val in samples:
            f.write(f"{val}\n")

    # Step 5: Read from temporary file, clean up, and rewrite final result
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


if __name__ == "__main__":
    for B in {2 ** 20}:
        for n in {2 ** 12, 2 ** 16, 2 ** 20, 2 ** 24}:
            alpha = 1.5  
            output_path = f"./Zip_n{n}B{B}"

            generate_zipf_data(n, B, alpha, output_path)
