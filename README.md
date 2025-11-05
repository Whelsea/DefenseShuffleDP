# DefenseShuffleDP

This repository contains the implementation of our unified framework for evaluating frequency-based queries under the shuffle differential privacy model. We evaluate three types of queries: **bit count**, **summation**, and **histogram**. Each query type is organized into a dedicated folder and supports multiple protocols including state-of-the-art baselines and our own designs.

---

## 📁 Project Structure

```
DefenseShuffleDP
├─count_sum
│  │  GKMPS.py
│  │  BBGN.py
│  │  SUSDP.py
│  │  BSDP.py
│  │  advanced_HSDP.py
│  │  run_experiments.py
│  │ 
│  └─data
│      └─Salary
│          ├─BR_Salaries
│          ├─Ont_Salaries
│          └─SF_Salaries
└─histogram
    │  FE1.py
    │  FE1_Simulator.py
    │  Flip_list.py
    │  ours_fe.py
    │  simulate_ours_fe.py
    │  run_experiments.py
    │  
    └─data
       ├─aol_data
       ├─BR_Salaries
       ├─Gauss
       ├─SF_Salaries
       └─Zip
```
---

## 🚀 How to Run

Each subdirectory includes a standalone experiment script:

- For `count` and `sum` queries:

  ```bash
  cd count_sum
  python run_experiments.py
  ```

  - All experimental settings for count and sum queries are configured in
    `count_sum/run_experiments.py`, inside the `main()` function.

  - To customize which protocols to run, modify the following list:

    ```python
    protocols = [
        "CSUZZ",
        "BBGN",
        "GKMPS",
        "ours+BBGN",
        "ours+GKMPS",
        "SUSDP+BBGN",
        "SUSDP+GKMPS",
        "BSDP+BBGN",
        "BSDP+GKMPS"
    ]
    ```

  - To adjust experiment parameters, modify the following lists:

    ```python
    list_num_users = [2 ** 16]           # Number of users
    list_domain = [2]                    # Data domain size
    list_k = [1]                         # Number of corrupted users
    list_attack_msg = [None]             # Number of messages per corrupted user sends (Fill None if no manual setting is required)
    list_epsilon = [1]                   # Privacy budget
    list_lambda = [256]                  # Bottom-group size (Fill None if no manual setting is required)
    list_dataset = ["Adult"]             # Dataset: "Adult", "SF_Salaries", etc.
    list_problem = ["Bit Counting"]      # Query type: "Bit Counting" or "Summation"
    list_distribution = ["Gauss"]        # Distribution type for synthetic data
    ```

  In addition, the running script also provides mathematically equivalent simulators (e.g., `"simulate SUSDP+BBGN"`, `"simulate SUSDP+GKMPS"`, and so on) that reproduce identical utility results as their full message-exchange implementations, offering a faster yet theoretically equivalent way for readers to verify our results.

  Note: When running **BSDP+BBGN**, BBGN computes `domain` with a factor dependent on group size (e.g. `self.domain = n * U * 10`), which causes modular wrapping that can “fold” the corrupted user’s `n` noisy messages and reduce the effective attack magnitude. To avoid underestimating attack strength, increase the multiplier used to build `domain` (for example, from `10` to `1e3` or `1e4`) in `BBGN.py`.

- For `histogram` queries:

  ```bash
  cd histogram
  python run_experiments.py
  ```

  - All settings can be modified at the beginning of `histogram/run_experiments.py`, in a similar manner to `count_sum`, by adjusting the `algorithms`, `data_modes`, and parameter lists such as `list_n`, `list_B`, `list_lambda`, etc.
  - For quick testing or debugging, the algorithms can also be executed directly within their corresponding files (e.g., `FE1.py` or `ours_fe.py`) without running the full experiment script.


The data/ directories include preprocessed samples for selected settings. Due to file size constraints, we do not include all datasets or parameter combinations. However, each dataset folder includes scripts for generating or processing the original data.

---
## 🧪 Supported Methods

### Count / Sum Queries

We evaluate the following protocols under the shuffle DP model for `count` and `sum` queries:

- **GKMPS**  
  *File:* `count_sum/GKMPS.py`  
- **BBGN**  
  *File:* `count_sum/BBGN.py`  
- **CSUZZ and our Framework (Ours+GKMPS, Ours+BBGN)**  
  *File:* `count_sum/advanced_HSDP.py`
  
### Histogram Queries

We evaluate the following protocols for `histogram` (i.e., frequency estimation) queries:

- **FE1 (LWY)**  
  *File:* `histogram/FE1.py`  
- **PFLIP (CZ)**  
  *File:* `histogram/Flip_list.py`  
- **Ours+FE1（Ours+LWY）**  
  *File:* `histogram/ours_fe.py`

For faster verification, equivalent simulators are also provided (`FE1_Simulator.py`, `simulate_ours_fe.py`).

---
## Dataset
We evaluate both synthetic and real-world datasets:

### Synthetic Datasets
We simulate data under three types of distributions: **Uniform**, **Zipfian**, **Gaussian**.

For count and sum queries, data generation is implemented in  
`count_sum/run_experiments.py` → `generate_data()`.

For histogram queries, pre-generated datasets are located under `histogram/data/`. (except uniform distribution)

### Real-world Datasets

| Dataset | Use Case | Description |
|--------|----------|-------------|
| [Adult Data](https://archive.ics.uci.edu/dataset/2/adult) | Count/Sum | Uses `sex` or `age` field |
| [SF Salaries](https://www.kaggle.com/datasets/kaggle/sf-salaries) | Count/Sum/Hist | Uses `BasePay` column |
| [Brazil Salaries](https://www.kaggle.com/datasets/gustavomodelli/monthly-salary-of-public-worker-in-brazil) | Count/Sum/Hist | Uses `total_salry` column |
| [AOL Dataset](http://www.cim.mcgill.ca/~dudek/206/Logs/AOL-user-ct-collection/) | Histogram | We use `user-ct-test-collection-01.txt` |
