# WSM project 2

## Quick Start

1. Install Poetry
   ```sh
   curl -sSL https://install.python-poetry.org | python3 -
   ```
2. Use Python 3.12 to run the project
   ```sh
   pyenv local 3.12  # if you have pyenv installed
   poetry env use $(command -v python3)  # make sure the right python is used
   ```
3. Install dependencies
   ```sh
   poetry install --no-root
   ```
4. Put in WT2G corpus in root and `proj2_sample_run/data` folder
5. Index the corpus
   ```sh
   ./indexing.sh
   ```
5. Run the programs. The relation of them is shown below:
   - Queries part 1: Run `search.py` with different parameters. Currently ran results are in `runs` folder.
   - Queries part 2: Run `queries_part2/train.py` to train the model. The trained model is in `model.pth`. The performance of the trained model will be calculated. Meanwhile, run `queries_part2/eval_part1.py` to evaluate the performance of those ranking functions in part 1.
   - Evaluation part 1: Run `search.py` with different parameters. Currently ran results are in `eval_part1` folder.
   - Evaluation part 2: Run `search.py` with different parameters to see the results from those three ranking functions. And then, run `eval_part2/eval.py` to calculate the performance of the model. Currently ran results are in `eval_part2` folder.