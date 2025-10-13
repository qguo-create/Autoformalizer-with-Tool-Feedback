# Autoformalizer with Tool Feedback (ATF)

Welcome to the Autoformalizer with Tool Feedback (ATF) repository! This project aims to enhance the process of autoformalization by incorporating syntactic and semantic validation tools, thereby improving the accuracy and reliability of formal statements generated from natural language mathematical problems.

## Overview

Autoformalization addresses the scarcity of data for Automated Theorem Proving (ATP) by translating mathematical problems from natural language into formal statements. ATF introduces a novel approach that integrates syntactic and consistency information as tools into the formalization process. By leveraging Lean 4 compilers for syntax corrections and employing a multi-LLMs-as-judge approach for consistency validation, ATF adaptively refines generated statements according to tool feedback, enhancing both syntactic validity and semantic consistency.

## Benchmark Performance
xxx

## Key Features

- **Tool Feedback Integration**: ATF uses syntactic and consistency validation tools to guide the formalization process, ensuring high-quality formal statements.
- **Cold-Start Phase**: Introduces basic tool usage with synthetic trajectories to familiarize the model with tool-driven revisions.
- **Expert Iteration Phase**: Enhances formalization capabilities through iterative refinement and expert feedback.
- **Direct Preference Optimization (DPO)**: Reduces ineffective revisions and encourages the model to complete formalization in fewer attempts.
- **Open-Source Dataset**: Numina-ATF, a dataset containing 750K synthetic formal statements, is provided to facilitate advancements in autoformalization and ATP research.

## Installation

To get started with ATF, clone this repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/Autoformalizer-with-Tool-Feedback.git
cd Autoformalizer-with-Tool-Feedback
pip install -r requirements.txt
```

## Usage

The repository includes scripts and configurations for training the ATF model. You can start training by running the following command:

```bash
python train.py --config config.yaml
```

For inference, use the following command:

```bash
python infer.py --input your_input_file.txt --output your_output_file.txt
```

## Dataset

The Numina-ATF dataset is available for download [here](link-to-dataset). This dataset includes 750K formal statements derived from Numina-V1.5 queries, validated for both syntax and consistency.

## Contributing

We welcome contributions to improve ATF. Please feel free to submit issues, feature requests, or pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please contact [
