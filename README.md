# Robustness Testing and Training Against Perturbations

## Overview
This project focuses on evaluating and improving the robustness of machine learning models against input perturbations.
The code generates datasets with noise, trains models, and evaluates their robustness using specific metrics.

## Main Components

### Data Generation
- **NoiseGenerator**: Generates different types of noise (normal, uniform, laplace, etc.) to perturb the data.
- **DatasetGenerator**: Creates datasets based on a given equation and applies noise to simulate real-world perturbations.

### Model Training
- **ModelTrainer**: Handles the training of different types of models (e.g., linear models, CNNs, random forests) with specified loss functions and training configurations.

### Evaluation
- **RobustnessMetric**: Calculates robustness metrics to evaluate the performance of models under noisy conditions.

## Usage

1. **Install requirements**: Install the requirements file using the command:
```bash
cd code
pip install -r requirements.txt
```
2. **Configuration**: Default configurations for a subset of equations are set in `config/equations`.
Each equation has a single JSON file that defines the model type, training type, number of epochs, early stopping criteria, indices of target features for perturbation, and other relevant parameters.

3. **Run the Main Script**: Execute the `main.py` script will perform the training process for a set of equations using MLP model with Mean Squared Error as the loss function.

```bash
python main.py
```
For running the main script using `msep` as a loss function, run:
```bash
python main.py --loss_function=msep
```

4. **View Results**: After running the main script, the results of the training, including metrics and model performance, will be saved in the specified output directory. You can analyze these results to understand the robustness of your models.

## Generating the Full Set of Equations from AI-Feynman

To generate the full set of equations with default values, run the `excel_parse.py` script located in the `utils` directory. This script will parse an Excel file containing the equations and generate the corresponding JSON configuration files.

```bash
cd utils/
python excel_parse.py
```

Make sure the Excel file, which can be downloaded from [AI-Feynman](https://space.mit.edu/home/tegmark/aifeynman.html), is properly formatted and located in the expected directory as specified in the script.
