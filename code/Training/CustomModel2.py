import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from Training.Models import BaseModel
from Metric.RobustnessMetric import RobustnessMetric
class CustomModel(BaseModel):
    def __init__(self, input_shape, loss_function, output_shape=1):
        super().__init__(loss_function)
        self.loss_function = tf.keras.losses.MeanSquaredError()
        # Define the layers
        self.dense1 = layers.Dense(64, activation='relu', input_shape=[input_shape])
        self.dense2 = layers.Dense(64, activation='relu')
        self.dense3 = layers.Dense(output_shape)
        # Define the metrics
        self.train_loss = tf.keras.metrics.MeanSquaredError(name='train_loss')
        self.valid_loss = tf.keras.metrics.MeanSquaredError(name='valid_loss')
        self.optimizer = tf.keras.optimizers.Adam()
        self.metric = RobustnessMetric()
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

    def compile_and_fit(self, xy_train, xy_valid, bl_training_ratio, bl_validation_ratio, xy_noisy, xy_clean, gx_gy, indices, fit_args, weights):
        print("bl_training_ratio: ", bl_training_ratio)

        fit_args_copy = fit_args.copy()
        bl_training_ratio = tf.constant(bl_training_ratio)
        bl_training_ratio = tf.cast(bl_training_ratio, tf.float32)
        bl_validation_ratio = tf.constant(bl_validation_ratio)
        bl_validation_ratio = tf.cast(bl_validation_ratio, tf.float32)
        penalty = 0
        x_noisy, y_noisy = xy_noisy
        
        x_clean, y_clean = xy_clean
        gx, gy = gx_gy
        
        indices_train, indices_valid = indices
        # x_noisy_train = x_noisy[:, indices_train, :]
        # x_noisy_valid = x_noisy[:, indices_valid, :]

        # x_noisy_train_max_bounds = np.zeros((x_noisy_train.shape[1], x_noisy_train.shape[-1]))
        # x_noisy_train_min_bounds = np.zeros((x_noisy_train.shape[1], x_noisy_train.shape[-1]))
        # x_noisy_valid_max_bounds = np.zeros((x_noisy_valid.shape[1], x_noisy_valid.shape[-1]))
        # x_noisy_valid_min_bounds = np.zeros((x_noisy_valid.shape[1], x_noisy_valid.shape[-1]))
    
        # for noisy_feat in range(x_noisy_train.shape[-1]):
        #     x_noisy_train_max_bounds[:, noisy_feat], x_noisy_train_min_bounds[:, noisy_feat] = self.metric.extract_bounds(x_noisy_train[:, :, noisy_feat])
        #     x_noisy_valid_max_bounds[:, noisy_feat], x_noisy_valid_min_bounds[:, noisy_feat] = self.metric.extract_bounds(x_noisy_valid[:, :, noisy_feat])
        
        
        y_noisy_train = y_noisy[:, indices_train, ]
        
        x_clean_train = x_clean[indices_train, :]
        x_train_bounds = self.metric.extract_bounds(x_clean_train)
        
        x_clean_valid = x_clean[indices_valid, :]
        y_clean_train = y_clean[indices_train,]
        y_clean_valid = y_clean[indices_valid,]
        history = {'loss': []}
        training_constant = tf.constant(bl_training_ratio)
        valid_constant = tf.constant(bl_validation_ratio)
        input_features = xy_train[0].shape[1]       
        xy_train = tf.data.Dataset.from_tensor_slices(xy_train).batch(xy_train[0].shape[0])
        xy_valid = tf.data.Dataset.from_tensor_slices(xy_valid).batch(xy_valid[0].shape[0])
        history['val_loss'] = []
        
        best_valid_loss = np.inf
        no_improvement = 0
        # TODO: read patience from fit_args
        patience = 20
        
        for epoch in range(fit_args_copy['epochs']):
            # Training loop
            for batch in xy_train:
                inputs, targets = batch
                with tf.GradientTape() as tape:
                    predictions = self(inputs)
                    loss = self.loss_function(targets, predictions) + penalty
                    y_noisy = self(x_noisy)

                    # loss = self.loss_function(targets, predictions)
                gradients = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                outer_dists = ["Euclidean"]
                training_ratio = self.metric.calculate_metric(x=inputs.numpy(), y=targets.numpy(),
                                                     x_hat=inputs.numpy(), y_hat=predictions.numpy(),
                                                     outer_dist=outer_dists, weights=weights, save=False, vis=False)

                # training_ratio = self.metric.calculate_metric(x=inputs.numpy(), y=targets.numpy(),
                #                                      x_bounds=(x_noisy_train_max_bounds, x_noisy_train_min_bounds), y_hat=y_noisy_train.numpy(),
                #                                      outer_dist=outer_dists, weights=weights, save=False, vis=False)
                training_input = tf.constant(training_ratio[outer_dists[0]]["Input distance"])
                training_output = tf.constant(training_ratio[outer_dists[0]]["Output distance"])
                training_ratio = tf.constant(training_ratio[outer_dists[0]]["Ratio"])
                # convert to float 
                training_ratio = tf.cast(training_ratio, tf.float32)
                print("training_ratio: ", training_ratio)
                # difference between the training ratio and the baseline training ratio
                diff = tf.math.subtract(training_constant, training_ratio)
                penalty = tf.maximum(tf.constant(0.), diff)    
                            
                # Update the metrics
                self.train_loss.update_state(targets, predictions)                        
            # Validation loop
            for valid_batch in xy_valid:
                valid_inputs, valid_targets = valid_batch
                valid_predictions = self(valid_inputs)
                # valid_y_noisy = self(x_noisy_valid)
                # valid_ratio = self.metric.calculate_metric(x=x_clean_valid, y=y_clean_valid,
                #                                            x_bounds=(x_noisy_valid_max_bounds, x_noisy_valid_min_bounds), y_hat=valid_y_noisy.numpy(),
                #                                            outer_dist=outer_dists, weights=weights, save=False, vis=False)
                # vaid_input = tf.constant(valid_ratio[outer_dists[0]]["Input distance"])
                # valid_output = tf.constant(valid_ratio[outer_dists[0]]["Output distance"])
                # valid_ratio = tf.constant(valid_ratio[outer_dists[0]]["Ratio"])
                # valid_ratio = tf.cast(valid_ratio, tf.float32)
                # valid_penalty = tf.maximum(0., valid_constant - valid_ratio)
                
                # valid_loss = self.loss_function(valid_targets, valid_predictions) + valid_penalty
                valid_loss = self.loss_function(valid_targets, valid_predictions)
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    no_improvement = 0
                else:
                    no_improvement += 1
                    
                self.valid_loss.update_state(valid_targets, valid_predictions)
            # Display the metrics at the end of each epoch
            template = 'Epoch {}, Loss: {}, Valid Loss: {}, Training Ratio: {}, training penalty: {}, training input: {}, training output: {}, training constant: {}'
            print(template.format(epoch+1,
                                  self.train_loss.result(),
                                  self.valid_loss.result(),
                                    training_ratio,
                                    penalty,
                                    training_input, training_output, training_constant))
            
            # Store the metrics in the history
            # Check for early stopping
            if no_improvement >= patience:
                print(f'Early stopping after {epoch} epochs')
                break
            # save the last number of epochs in the history
            history['val_loss'].append(self.valid_loss.result().numpy())
            history['train_ratio'] = training_ratio
            # history['valid_ratio'] = valid_ratio
            history["epoch"] = epoch
            
            history['loss'].append(self.train_loss.result().numpy())
            # history['val_loss'].append(self.valid_loss.result().numpy())
            # Reset the metrics for the next epoch
            self.train_loss.reset_states()
            self.valid_loss.reset_states()
        ratio = self.metric.calculate_metric(x=x_clean, y=y_clean,
                                                        x_hat=x_noisy, y_hat=y_noisy.numpy(),
                                                        outer_dist=outer_dists, weights=weights, save=False, vis=False)
        print("ratio: ", ratio)
        exit()
        return self, history
    # def save_model(self, path):
    #     tf.saved_model.save(self, path)
    
    def evaluate_and_plot(self, x_test, y_test, path="./"):
        # Make predictions on the test set
        y_pred = self.call(x_test)

        # Calculate the loss on the test set
        test_loss = self.loss_function(y_test, y_pred)
        print(f'Test loss: {test_loss}')

        # Plot the predicted results against the actual results
        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title('Actual vs Predicted Results')
        plt.legend()
        plt.show()
        
    # def load_model(self, path):
    #     model = tf.keras.models.load_model(path)
    #     return model
    # def save_model(self, path):
    #     self.save_weights(path)

    def save_model(self, path):
        tf.keras.models.save_model(self, path)
    
    def load_model(self, path):
        model = tf.keras.models.load_model(path)
        return model
 

    
    # @classmethod
    # def load_model(cls, path, *args, **kwargs):
    #     model = cls(*args, **kwargs)  # Initialize the model with the constructor arguments
    #     model.load_weights(path)  # Load the weights
    #     return model
# if __name__ == "__main__":
#     import numpy as np
#     import tensorflow as tf
#     from tensorflow.data import Dataset

#     # Create a synthetic dataset for a regression problem
#     x_train = np.random.rand(1000, 784).astype(np.float32)
#     y_train = np.random.rand(1000, 1).astype(np.float32)
#     x_valid = np.random.rand(200, 784).astype(np.float32)
#     y_valid = np.random.rand(200, 1).astype(np.float32)

#     # Convert the datasets to TensorFlow Datasets
#     xy_train = Dataset.from_tensor_slices((x_train, y_train)).batch(32)
#     xy_valid = Dataset.from_tensor_slices((x_valid, y_valid)).batch(32)

#     # Define the loss function
#     loss_function = tf.keras.losses.MeanSquaredError()

#     # Create an instance of CustomModel
#     model = CustomModel(input_shape=784, loss_function=loss_function, output_shape=1)

#     # Define the fit arguments
#     fit_args = {'epochs': 10}

#     # Train the model
#     model, history = model.compile_and_fit(xy_train, xy_valid, fit_args)

#     # Print the training history
#     print(history)
