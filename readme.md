# Echo / Echo-Think Project

This repository implements the training and evaluation pipelines for **Echo** and **Echo-Think (Hybrid)** models.

## 🚀 Quick Start

There are two modes available: **Standard Echo** and **Think Hybrid**. Please follow the corresponding instructions below.

### 1. Standard Echo Mode

If you wish to use the standard Echo implementation:

**Step 1: Training**
Run the following script to start training:
```bash
bash run_training.sh

```


**Step 2: Evaluation Once training is finished, run the evaluation:**
```bash
bash run_longbench.sh

```

### 2. Echo with Think Hybrid

**Step 1: Training**
Run the following script to start training:
```bash
bash run_training_think.sh

```


**Step 2: Evaluation Once training is finished, run the evaluation:**
```bash
bash run_longbench_think.sh

```


## Dataset Note
For the sake of portability and ease of demonstration, this repository currently only includes the narrativeqa dataset from the LongBench suite.