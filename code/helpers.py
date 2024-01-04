import numpy as np
import matplotlib.pyplot as plt

# Helper function to get the data based on the type
def get_data(data_types, x_clean, y_clean, gx, gy):
    """
    Concatenates the data based on the specified data types.

    Parameters:
    data_types (list): A list of data types to include in the concatenated data.
    x_clean (ndarray): The clean x data.
    y_clean (ndarray): The clean y data.
    gx (ndarray): The gx data.
    gy (ndarray): The gy data.

    Returns:
    ndarray: The concatenated x data.
    ndarray: The concatenated y data.
    """
    x_data, y_data = [], []
    for data_type in data_types:
        if data_type == 'clean':
            x_data.append(x_clean)
            y_data.append(y_clean)
        elif data_type == 'gx':
            x_data.append(gx)
            y_data.append(gy)
    return np.concatenate(x_data), np.concatenate(y_data)


def evaluate(model, x):
    """
    Evaluate the model on the given data.

    Parameters:
    model (object): The trained model.
    x (ndarray): The input data.

    Returns:
    ndarray: The predicted values.
    """
    return model.predict(x)

def evaluate_and_plot(model, history, xy_test, path="./"):
    """
    Evaluate the model on the test data, generate predictions, and plot the predicted vs actual values and loss history.

    Parameters:
    model (object): The trained model.
    history (object): The training history of the model.
    xy_test (tuple): A tuple containing the test data (x_test, y_test).
    path (str): The path to save the generated plots. Default is the current directory.

    Returns:
    None
    """
    # Evaluate the model on the test data
    x_test = xy_test[0]
    y_test = xy_test[1]
    
    test_loss = model.evaluate(x_test, y_test)
    # Generate predictions on the test data
    y_pred = model.predict(x_test)

    # Plot the predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(-3, 3, len(y_test)), y_test, label='Actual')
    plt.plot(np.linspace(-3, 3, len(y_pred)), y_pred, label='Predicted')
    plt.title('Predicted vs Actual Values')
    plt.legend()
    plt.savefig(f"{path}/predicted_vs_actual.png", bbox_inches='tight', dpi=300)
    if history is not None:
        # Print the loss history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss History')
        plt.legend()
        plt.savefig(f"{path}/loss_history.png", bbox_inches='tight', dpi=300)