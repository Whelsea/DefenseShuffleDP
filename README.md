# DefenseShuffleDP

This repository contains the implementation of our unified framework for evaluating frequency-based queries under the shuffle differential privacy model. We evaluate three types of queries: **bit countint**, **summation**, and **histogram**. Each query type is organized into a dedicated folder and supports multiple protocols including state-of-the-art baselines and our own designs.

---

## 📁 Project Structure

```
DefenseShuffleDP
├─count_sum
│  │  advanced_HSDP.py
│  │  BBGN.py
│  │  GKMPS.py
│  │  run_experiments.py
│  │ 
│  └─data
│      └─Salary
│          ├─BR_Salaries
│          ├─Ont_Salaries
│          └─SF_Salaries
└─histogram
    │  FE1_Simulator.py
    │  Flip_list.py
    │  run_experiments.py
    │  simulate_ours_fe.py
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

- For `histogram` queries:
  ```bash
  cd histogram
  python run_experiments.py

The data/ directories include preprocessed samples for selected settings. Due to file size constraints, we do not include all datasets or parameter combinations. However, each dataset folder includes scripts for generating or processing the original data.


## 🧪 Supported Methods

### Count / Sum Queries

We evaluate the following protocols under the shuffle DP model for `count` and `sum` queries:

- **GKMPS**  
  *File:* `count_sum/GKMPS.py`  
- **BBGN**  
  *File:* `count_sum/BBGN.py`  
- **Our Framework (CSUZZ, Ours+GKMPS， Ours+BBGN)**  
  *File:* `count_sum/advanced_HSDP.py`
  
### Histogram Queries

We evaluate the following protocols for `histogram` (i.e., frequency estimation) queries:

- **FE1 (LWY)**  
  *File:* `histogram/FE1_Simulator.py`  
- **PFLIP (CZ)**  
  *File:* `histogram/Flip_list.py`  
- **Ours+FE1（Ours+LWY）**  
  *File:* `histogram/simulate_ours_fe.py`  


## Dataset
