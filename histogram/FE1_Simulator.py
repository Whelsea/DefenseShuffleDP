import numpy as np
import math
import time
import random
from typing import List
import multiprocessing
from functools import partial
import os


class FE1Simulator:
    def __init__(self, n: int, B: int, epsilon: float, delta: float, c: float, beta: float):
        self.n = n
        self.B = B
        self.epsilon = epsilon
        self.delta = delta
        self.c = c
        self.beta = beta
        # self.b = int(n / math.pow(math.log(n), c))
        self.b = int(max(1, epsilon ** 2 * n / math.pow(math.log(n), c)))
        print("b=", self.b)
        self.q = max(B, self.b) + 1
        while not self._is_prime(self.q):
            self.q += 1

        self.mu = self._search_mu()
        print(f"n={n},epsilon = {epsilon},delta={delta}")
        print(f"mu={self.mu}")
        print(f"b={self.b}")
        print(
            f"Bits per message: {math.ceil(math.log2(self.q)) * 2 + math.ceil(math.log2(self.b))}")
        print(math.log2(n))
        print(
            f"Bits per message: {math.ceil(math.log2(self.q)) * 2 + math.ceil(math.log2(self.b)) + math.log2(self.n)}")
        self.sample_prob = self.mu * self.b / self.n
        print(f"Msg:{1 + self.sample_prob}")
        self.pcol = (self.q // self.b) * (self.q % self.b + self.q - self.b) / (self.q * (self.q - 1))

    def get_theta_fe1(self):
        """
        计算FE1算法在给定参数下的误差上界theta (计算最坏情况下的theta)

        返回:
        theta: 误差上界
        """
        b = self.b
        pcol = self.pcol
        rho = self.sample_prob

        # 计算μ (噪声Y的期望)
        mu_noise1 = self.n * pcol
        mu_noise2 = (self.n * 2 * math.floor(rho)) * (1 / b)
        mu_noise3 = (self.n * 2) * ((rho - math.floor(rho)) / b)
        mu = mu_noise1 + mu_noise2 + mu_noise3

        # 计算 (误差上界)
        term1 = 3 * math.log(2 * self.B / self.beta)
        term2 = math.sqrt(3 * math.log(2 * self.B / self.beta) * mu) / (1 - pcol)

        # 这个bias是实际加入的噪声和直接按照2n去偏产生的，现在先设置为0
        bias = 0
        theta = max(term1, term2) + bias

        return theta

    def fast_binomial_noise(self, n, p):
        """使用正态分布近似二项分布，更快速"""
        if n * p * (1 - p) < 9:  # 当 np(1-p) 较小时使用精确计算
            return np.random.binomial(n, p)
        else:
            # 正态分布近似: N(np, np(1-p))
            mu = n * p
            sigma = math.sqrt(n * p * (1 - p))
            return int(round(np.random.normal(mu, sigma)))

    def process_batch(self, indices, g, n, b, pcol, rho, honest_user_proportion):
        """处理一批索引，返回它们的估计频率"""
        results = np.zeros(len(indices))

        # 预计算不依赖于具体元素的噪声部分

        for i, x in enumerate(indices):
            gx = g[x]

            noise1 = np.random.binomial(max(0, n * honest_user_proportion - int(gx)), pcol)
            noise2 = np.random.binomial(int(n * honest_user_proportion * math.floor(rho)), 1 / b)
            noise3 = np.random.binomial(n * honest_user_proportion, (rho - math.floor(rho)) / b)

            X = int(gx) + noise1 + noise2 + noise3
            gx_hat = (X - n * rho / b - n * pcol) / (1 - pcol)
            results[i] = gx_hat

        return results

    def simulate_parallel(self, data: List[int], honest_user_proportion) -> np.ndarray:
        """使用并行处理加速模拟所有B个元素"""
        # 计算真实频率
        g = np.zeros(self.B + 1)
        for x in data:
            g[x] += 1

        est = np.zeros(self.B + 1)
        n = self.n
        b = self.b
        pcol = self.pcol
        rho = self.sample_prob

        # 处理所有元素，但通过分块提高效率
        indices_to_process = np.arange(1, self.B + 1)
        num_cores = multiprocessing.cpu_count()
        indices_splits = np.array_split(indices_to_process, num_cores)

        # 创建进程池并执行并行处理
        with multiprocessing.Pool(processes=num_cores) as pool:
            process_func = partial(self.process_batch, g=g, n=n, b=b, pcol=pcol, rho=rho,
                                   honest_user_proportion=honest_user_proportion)
            results = pool.map(process_func, indices_splits)

        # 合并结果
        for i, batch_indices in enumerate(indices_splits):
            est[batch_indices] = results[i]

        return est  # skip index 0

    # simulator
    def simulate(self, data: List[int]) -> np.ndarray:
        g = np.zeros(self.B + 1)
        for x in data:
            g[x] += 1

        est = np.zeros(self.B + 1)

        n = self.n
        b = self.b
        pcol = self.pcol
        rho = self.sample_prob

        for x in range(1, self.B + 1):
            gx = g[x]
            noise1 = np.random.binomial(n - int(gx), pcol)
            noise2 = np.random.binomial(int(n * math.floor(rho)), 1 / b)
            noise3 = np.random.binomial(n, (rho - math.floor(rho)) / b)
            X = int(gx) + noise1 + noise2 + noise3
            gx_hat = (X - n * rho / b - n * pcol) / (1 - pcol)
            est[x] = gx_hat

        return est  # skip index 0

    def _is_prime(self, p: int) -> bool:
        test_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        for tp in test_primes:
            if p % tp == 0 or pow(tp, p - 1, p) != 1:
                return False
        return True

    def _search_mu(self) -> float:
        epow = math.exp(self.epsilon)
        le, ri = 0.0, 1000.0 / self.n

        def checker(p):
            prob = [0.0] * (self.n + 1)
            accprob = [0.0] * (self.n + 1)
            C = 1.0
            for i in range(self.n + 1):
                prob[i] = C * pow(1 - p, self.n - i)
                if i < self.n:
                    C = C * (self.n - i) * p / (i + 1)
            accprob[self.n] = prob[self.n]
            for i in reversed(range(self.n)):
                accprob[i] = accprob[i + 1] + prob[i]
            pro = 0.0
            for x2 in range(self.n + 1):
                x1 = math.ceil(epow * x2 - 1)
                if x1 >= self.n:
                    break
                x1 = max(0, x1)
                pro += prob[x2] * accprob[x1]
            return pro <= self.delta

        while le + 0.1 / self.n < ri:
            mi = (le + ri) / 2
            if checker(mi):
                ri = mi
            else:
                le = mi
        # return ri * self.n
        print(2 / self.delta)
        print(2 * self.n * self.n)
        print(math.log(2 / self.delta))
        print((self.epsilon ** 2))
        print(32 * math.log(2 / self.delta) / (self.epsilon ** 2))
        return 32 * math.log(2 / self.delta) / (self.epsilon ** 2)


def loaddata(path: str) -> List[int]:
    with open(path, 'r') as f:
        n = int(f.readline())
        B = int(f.readline())
        return [int(f.readline()) for _ in range(n)], n, B

def load_data_by_mode(data_mode: str, n: int, B: int):
    if data_mode == "zipf":
        path = f"./data/Zip/Zip_n{n}B{B}"
        data, _, _ = loaddata(path)
    elif data_mode == "gauss":
        path = f"./data/Gauss/Gauss_n{n}B{B}"
        data, _, _ = loaddata(path)
    elif data_mode == "aol":
        path = f"./data/Real/aol_n{n}B{B}"
        data, _, _ = loaddata(path)
    elif data_mode == "unif":
        data = np.random.randint(0, B, size=n)
    else:
        raise ValueError(f"Unsupported data_mode: {data_mode}")
    return data


if __name__ == "__main__":
    # 实验参数
    n = 131072
    B = 131072
    epsilon = 4.0
    c = 1.0
    delta = 1.0 / (n * n)
    beta = 0.1
    num_runs = 50
    trim_count = 5

    list_distribution = ["unif", "zipf", "gauss","aol"]  # 遍历多个模拟分布

    for data_mode in list_distribution:
        print(f"\n====== Running {data_mode.upper()} ======")

        data = load_data_by_mode(data_mode, n, B)
        sim = FE1Simulator(n, B, epsilon, delta, c, beta)

        true_freq = np.zeros(B + 1)
        for x in data:
            true_freq[x] += 1

        parallel_results = {'runtime': [], 'max_error': [], 'msg_count': [], 'bit_count': []}
        serial_results = {'runtime': [], 'max_error': []}

        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}")

            # 并行模拟
            start_time = time.time()
            est_freq_parallel = sim.simulate_parallel(data, 1)
            parallel_time = time.time() - start_time
            errors = np.abs(true_freq[1:] - est_freq_parallel[1:])

            parallel_results['runtime'].append(parallel_time)
            parallel_results['max_error'].append(np.max(errors))
            parallel_results['msg_count'].append(1 + sim.sample_prob)
            parallel_results['bit_count'].append(math.ceil(math.log2(sim.q)) * 2 + math.ceil(math.log2(sim.b)))

            # 串行模拟
            start_time = time.time()
            est_freq_serial = sim.simulate(data)
            serial_time = time.time() - start_time
            errors = np.abs(true_freq[1:] - est_freq_serial[1:])

            serial_results['runtime'].append(serial_time)
            serial_results['max_error'].append(np.max(errors))

        def trimmed_mean(data, trim):
            sorted_data = sorted(data)
            return np.mean(sorted_data[trim:-trim] if trim > 0 else sorted_data)

        result_dir = os.path.join(".", "Result", "origin_FE1", data_mode.capitalize())
        os.makedirs(result_dir, exist_ok=True)
        result_file = os.path.join(result_dir, f"{data_mode}_n{n}B{B}_eps{epsilon}_reviseln.txt")

        with open(result_file, 'w') as f:
            f.write(f"=== FE1 Simulation Results ({data_mode.upper()} - n={n}, B={B}) ===\n")
            f.write(f"Total runs: {num_runs}\n")
            f.write(f"Trimmed runs: {num_runs - 2 * trim_count} (removed top/bottom {trim_count})\n\n")

            f.write("--- Parallel Simulation ---\n")
            f.write(f"mu: {sim.mu}\n")
            f.write(f"Runtime: {trimmed_mean(parallel_results['runtime'], trim_count):.4f} seconds\n")
            f.write(f"Max error: {trimmed_mean(parallel_results['max_error'], trim_count):.4f}\n")
            f.write(f"Messages per user: {trimmed_mean(parallel_results['msg_count'], trim_count):.4f}\n")
            f.write(f"Bits per user: {trimmed_mean(parallel_results['bit_count'], trim_count):.4f}\n\n")

            f.write("--- Serial Simulation ---\n")
            f.write(f"Runtime: {trimmed_mean(serial_results['runtime'], trim_count):.4f} seconds\n")
            f.write(f"Max error: {trimmed_mean(serial_results['max_error'], trim_count):.4f}\n\n")

            speedup = trimmed_mean(serial_results['runtime'], trim_count) / trimmed_mean(parallel_results['runtime'], trim_count)
            f.write(f"Speedup: {speedup:.2f}x\n")

        print(f"Results saved to: {os.path.abspath(result_file)}")
