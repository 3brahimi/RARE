import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

def get_data_from_dict(config, x_clean, y_clean, gx, gx_y):
    """
    config (dict): A dictionary of data types to include in the concatenated data.
    x_clean (ndarray): The clean x data.
    y_clean (ndarray): The clean y data.
    gx (ndarray): The gx data.
    gy (ndarray): The gy data.
    returns:
    x_train (ndarray): The concatenated x data for training.
    y_train (ndarray): The concatenated y data for training.
    x_valid (ndarray): The concatenated x data for validation.
    y_valid (ndarray): The concatenated y data for validation.
    x_test (ndarray): The concatenated x data for testing.
    y_test (ndarray): The concatenated y data for testing.
    """
    # copy the config dictionary to avoid changing the original one
    data_dict = config.copy()
   
   
    # add indices column to x_clean, so that we can know which indices were used for training, validation, and testing
    indices = np.arange(x_clean.shape[0])
    
        
    # x_train, x_temp, y_train, y_temp = train_test_split(x_clean, y_clean, test_size=0.2, random_state=42, shuffle=False)
    indices_train, indices_temp, x_train, x_temp, y_train, y_temp = train_test_split(indices, x_clean, y_clean, test_size=0.2, random_state=42)
    
    # Split the temporary set into validation and testing sets
    # x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, shuffle=False
    # )
    indices_valid, indices_test, x_valid, x_test, y_valid, y_test = train_test_split(indices_temp, x_temp, y_temp, test_size=0.5, random_state=42)
    
    # ####### no shuffle split     
    # x_train, x_temp, y_train, y_temp = train_test_split(x_clean, y_clean, test_size=0.2, random_state=42, shuffle=False)
    # # Split the temporary set into validation and testing sets
    # x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)

    ###### the following split is for the old code with get_data, the last three commented lines in this function
    # x_train_clean, x_temp_clean, y_train_clean, y_temp_clean = train_test_split(x_clean, y_clean, test_size=0.1, random_state=42, shuffle=False)
    # print("x_train_clean shape is", x_train_clean.shape, "y_train_clean shape is", y_train_clean.shape, "x_valid shape is", x_valid.shape, "y_valid shape is", y_valid.shape, "x_test shape is", x_test.shape, "y_test shape is", y_test.shape)
    
    # if training contains gx, then add it as a new feature column for training dataset
          # remove gx from all elements of the dictionary
    for key in data_dict:
        if key == 'training' and 'gx' in data_dict[key]:
            # concatenate x_train with gx of training_indices as a new column feature
            x_train = np.concatenate((x_train, gx[indices_train]), axis=1)
            # now double the length of x_train by repeating it, setting the gx values to 0
            x_train_new = np.concatenate((x_train, x_train), axis=0)
            x_train_new[:, -gx.shape[1]:] = 0
            x_train = x_train_new
            # now repeat y_train as well
            y_train = np.concatenate((y_train, y_train), axis=0)
            # for the validation set, add new column feature for gx only with values 0
            gx_valid = np.zeros((x_valid.shape[0], gx.shape[1]))
            x_valid = np.concatenate((x_valid, gx_valid), axis=1)
            # append the last values of gx to the test set, from the end back the length of x_test
            x_test = np.concatenate((x_test, gx[indices_test]), axis=1)
            x_test = np.concatenate((x_test, x_test), axis=0)
            
            y_test = np.concatenate((y_test, y_test), axis=0)
            data_dict[key] = [x for x in data_dict[key] if x != 'gx']
            
                     
        else:
            gx_train, gx_temp, gy_train, gy_temp = train_test_split(gx, gx_y, test_size=0.2, random_state=42)
            gx_valid, gx_test, gy_valid, gy_test = train_test_split(gx_temp, gy_temp, test_size=0.5, random_state=42)
 
    # x_train, y_train = get_data(data_dict['training'], x_train_clean, y_train_clean, gx_train, gy_train)

    # x_valid, y_valid = get_data(data_dict['validation'], x_temp_clean, y_temp_clean, gx_valid, gy_valid)

    # x_test, y_test = get_data(data_dict['testing'], x_temp_clean, y_temp_clean, gx_test, gy_test)
    return x_train, y_train, x_valid, y_valid, x_test, y_test

def noise_aware_data(config, x_clean, y_clean, gx, gx_y):
    """
    config (dict): A dictionary of data types to include in the concatenated data.
    x_clean (ndarray): The clean x data.
    y_clean (ndarray): The clean y data.
    gx (ndarray): The gx data.
    gy (ndarray): The gy data.
    returns:
    x_train (ndarray): The concatenated x data for training.
    y_train (ndarray): The concatenated y data for training.
    x_valid (ndarray): The concatenated x data for validation.
    y_valid (ndarray): The concatenated y data for validation.
    x_test (ndarray): The concatenated x data for testing.
    y_test (ndarray): The concatenated y data for testing.
    """
    # copy the config dictionary to avoid changing the original one
    data_dict = config.copy()
   
    indices = np.arange(x_clean.shape[0])
         
    indices_train, indices_temp, x_train, x_temp, y_train, y_temp = train_test_split(indices, x_clean, y_clean, test_size=0.2, random_state=42, shuffle=True)
   
    indices_valid, indices_test, x_valid, x_test, y_valid, y_test = train_test_split(indices_temp, x_temp, y_temp, test_size=0.5, random_state=42, shuffle=True)
   
    # ####### no shuffle split     
    # x_train, x_temp, y_train, y_temp = train_test_split(x_clean, y_clean, test_size=0.2, random_state=42, shuffle=False)
    # # Split the temporary set into validation and testing sets
    # x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, shuffle=False)

    
    # # creat gx_training, which is gx with the same length as x_train, and zeros for another training size
    # gx_train = np.concatenate((gx[indices_train], np.zeros((x_train.shape[0], gx.shape[1]))), axis=0)
    
    # copy x instead of zeros
    gx_train = np.concatenate((gx[indices_train], x_clean[indices_train]), axis=0)
    x_train_new = np.concatenate((x_train, x_train), axis=0)
    
    x_train = np.concatenate((x_train_new, gx_train), axis=1)
   
    # ## also gx_train - x_train as a new feature
    # dist_gx_x_train = gx_train - x_train_new
    # x_train = np.concatenate((x_train_new, gx_train, dist_gx_x_train), axis=1)
 
       
    # repeat y_train as well
    y_train = np.concatenate((y_train, y_train), axis=0)

    # for the validation set, add new column feature for gx only with values 0
    # gx_valid = np.zeros((x_valid.shape[0], gx.shape[1]))
    
    #  valid on clean data not zeros 
    gx_valid = x_valid
    # gx_valid = np.zeros((x_valid.shape[0], gx.shape[1]))
    # dist_gx_x_valid = gx_valid - x_valid
    x_valid = np.concatenate((x_valid, gx_valid), axis=1)
    # x_valid = np.concatenate((x_valid, gx_valid, dist_gx_x_valid), axis=1)
    
    # append the last values of gx to the test set, from the end back the length of x_test
    dist_gx_x_test = gx[indices_test] - x_test
    x_test = np.concatenate((x_test, gx[indices_test]), axis=1)    
    # x_test = np.concatenate((x_test, gx[indices_test], dist_gx_x_test), axis=1)  
    x_test = np.concatenate((x_test, x_test), axis=0)
    
    y_test = np.concatenate((y_test, y_test), axis=0)

    return x_train, y_train, x_valid, y_valid, x_test, y_test, indices_train, indices_valid


def noisy_data(x_clean, y_clean, x_noisy, y_noisy):
    x_all = None
    print("x_noisy shape is", x_noisy.shape, "y_noisy shape is", y_noisy.shape)
    print("x_clean shape is", x_clean.shape, "y_clean shape is", y_clean.shape)
    if x_noisy.shape[0] != x_clean.shape[0]:
        x_all = np.concatenate((x_clean, x_noisy.reshape(-1, x_noisy.shape[2])), axis=0)
    else:
        x_all = np.concatenate((x_clean, x_noisy), axis=0)
            
    y_all = np.concatenate((y_clean, y_noisy.reshape(-1)), axis=0)
    indices = np.arange(x_all.shape[0])
    indices_train, indices_temp,  x_train, x_temp, y_train, y_temp = train_test_split(indices, x_all, y_all, test_size=0.2, random_state=42, shuffle=True)
    indices_valid, indices_test, x_valid, x_test, y_valid, y_test = train_test_split(indices_temp, x_temp, y_temp, test_size=0.5, random_state=42, shuffle=True)
    return x_train,y_train,x_valid,y_valid,x_test,y_test,indices_train,indices_valid

def get_adversarial_data(config, x_clean, y_clean, gx, gx_y):
    """
    the adversarial data is split clean data for training and testing, and use gx for validation
    """
    indices = np.arange(x_clean.shape[0])
    indices_train, indices_test, x_train, x_test, y_train, y_test = train_test_split(indices, x_clean, y_clean, test_size=0.2, random_state=42)
    x_valid, x_test, y_valid, y_test = train_test_split(gx, gx_y, test_size=0.2, random_state=42)
    return x_train, y_train, x_valid, y_valid, x_test, y_test, indices_train, indices

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
    print("x_test shape: ", x_test.shape)
    print("y_test shape: ", y_test.shape)
    
    test_loss = model.evaluate(x_test, y_test)
    # Generate predictions on the test data
    y_pred = model.predict(x_test)
    if y_pred.shape[1] == 1:
        # Plot the predicted vs actual values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, label='Actual vs Predicted')
        # plt.plot(np.linspace(-3, 3, len(y_test)), y_test, label='Actual')
        # plt.plot(np.linspace(-3, 3, len(y_pred)), y_pred, label='Predicted')
        plt.title('Predicted vs Actual Values')
        plt.legend()
        plt.savefig(f"{path}/predicted_vs_actual.png", bbox_inches='tight', dpi=300)
    else:
        fig, ax = plt.subplots(figsize=(10, 6), nrows=y_pred.shape[1], ncols=1)
        for i in range(y_pred.shape[1]):
            # create outputs plot as subplots
            ax[i].scatter(y_test[:, i], y_pred[:, i])
            p1 = max(max(y_pred[:, i]), max(y_test[:, i]))
            p2 = min(min(y_pred[:, i]), min(y_test[:, i]))
            ax[i].plot([p1, p2], [p1, p2], 'k-')
            ax[i].set_title(f"Predicted vs Actual Values for output {i}")
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        # plt.legend()
        plt.savefig(f"{path}/predicted_vs_actual.png", bbox_inches='tight', dpi=300)
    if history is not None:
        # Print the loss history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss History')
        plt.legend()
        plt.savefig(f"{path}/loss_history.png", bbox_inches='tight', dpi=300)
        