# Ceivo ML Testing

This repository provides tools and scripts for evaluating and comparing multimodal AI models (text + image) using both AWS Bedrock and OpenAI APIs. It is designed to analyze single video frames, generate descriptive scene analyses, and benchmark model outputs against a baseline.

## Features
- **Multimodal Model Evaluation:**
  - Supports AWS Bedrock and OpenAI models for image + text analysis.
  - Compares model outputs with a baseline for benchmarking.
- **Batch Processing:**
  - Processes all images in a folder and saves results to `output.json`.
- **Jupyter Notebook Integration:**
  - Interactive analysis and visualization using `modelTesting.ipynb`.
- **Customizable Prompts:**
  - Easily modify prompts for different analysis requirements.

## File Overview
- `modelTesting.py`: Main script for batch processing images and evaluating models.
- `modelTesting.ipynb`: Jupyter notebook for interactive testing and visualization.
- `requirements.txt`: Python dependencies for the project.

## Setup
1. **Clone the repository:**
   ```pwsh
   git clone <repo-url>
   cd ceivo-ml-testing
   ```
2. **Install dependencies:**
   ```pwsh
   pip install -r requirements.txt
   ```
3. **Configure API Keys:**
   - Set your AWS Bedrock and OpenAI credentials as environment variables.
   - Example for AWS Bedrock:
     ```pwsh
     $env:AWS_BEARER_TOKEN_BEDROCK = "<your-bedrock-token>"
     ```

## Usage
### Batch Model Testing
Run the main script to process all images in a folder:
```pwsh
python modelTesting.py --folder <path-to-image-folder>
```
Results will be saved to `output.json`.

### Interactive Notebook
Open `modelTesting.ipynb` in Jupyter or VS Code for step-by-step analysis and visualization.

## Customization
- **Prompts:**
  - Edit the prompt in `modelTesting.py` or the notebook to change the analysis instructions.
- **Model List:**
  - Modify the `model_ids` list to add or remove models for evaluation.

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies.


