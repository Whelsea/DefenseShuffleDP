import numpy as np
import random
import os


def generate_gaussian_data(n, B, mu, sigma, output_path):
    """
    Generate Gaussian-distributed data by first shuffling [1, B], 
    then sampling indices using a Gaussian distribution
    """
    # Step 1: Randomly shuffle [1, B) as the value domain
    domain = list(range(1, B))
    random.shuffle(domain)

    # Step 2: Create a Gaussian distribution over B values and normalize
    x = np.arange(1, B)
    weights = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    prob = weights / weights.sum()

    # Step 3: Sample indices according to the probability distribution
    indices = np.random.choice(B - 1, size=n, replace=True, p=prob)
    samples = [domain[i] for i in indices]

    # Step 4: Write to a temporary file (to prevent data loss on interruption)
    temp_path = output_path + ".tmp"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(f"{n}\n")
        f.write(f"{B}\n")
        for val in samples:
            f.write(f"{val}\n")

    # Step 5: Clean up and rewrite valid entries
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


if __name__ == "__main__":
    for B in {131072}:  
        for n in {131072}:  
            mu = B // 5
            sigma = B / 5
            output_path = f"./Gauss_n{n}B{B}"

            generate_gaussian_data(n, B, mu, sigma, output_path)
