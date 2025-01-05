# LMCocktail

This repository introduces **LMCocktail**, a technique for data-aware model merging, as part of our research on advanced language model techniques. The implementation of LMCocktail can be found under [FlagEmbedding/research/LM_Cocktail](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/LM_Cocktail).

## Getting Started

### Prerequisites
1. Clone the repository:
   ```bash
   git clone https://github.com/FlagOpen/FlagEmbedding.git
   ```
2. Navigate to the `LM_Cocktail` directory and set it up:
   ```bash
   cd FlagEmbedding/research/LM_Cocktail
   ```

### Running the Code
To perform model merging with LMCocktail, run the following script:

```bash
python lmcocktail_merge_align.py --all_types --temp 1.0 --max_per_task 5
```

### Configuration Update
Before running the script, make sure to modify the function `preprocess_data_for_llm` in `LM_Cocktail/utils.py` to handle certain models (e.g., LLaMA-3). Update the code as follows:

```python
def preprocess_data_for_llm(example_data, tokenizer, device, batch_size:int=2, max_input_length:int=2048):
    batch_input_ids = []
    batch_labels = []
    batch_max_length = max_input_length
    tokenizer.pad_token_id = tokenizer.eos_token_id  # Add this line
```

## Example Usage

Here’s an example command to merge models using LMCocktail:

```bash
python lmcocktail_merge_align.py --all_types --temp 1.0 --max_per_task 5
```

### Parameters
- `--all_types`: Enables all types of model merging.
- `--temp`: Temperature for sampling. Default is `1.0`.
- `--max_per_task`: Maximum tasks to process per run. Default is `5`.
