import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from Training.Models import BaseModel
from Metric.RobustnessMetric import RobustnessMetric
class CustomModel(BaseModel):
    def __init__(self, input_shape, loss_function, output_shape=1):
        super().__init__(loss_function)
        self.shape_input = input_shape
        self.shape_output = output_shape
        self.model = self.model_architecture()
        if loss_function == 'mean_squared_error' or loss_function == 'mse':
            self.loss_function = tf.keras.losses.MeanSquaredError()
        # Define the metrics
        self.train_loss = tf.keras.metrics.MeanSquaredError(name='train_loss')
        self.valid_loss = tf.keras.metrics.MeanSquaredError(name='valid_loss')
        self.optimizer = tf.keras.optimizers.Adam()
        self.metric = RobustnessMetric()
    
    def model_architecture(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(16, input_shape=[self.shape_input], activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.shape_output)
        ])
        return model

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)
    
    def compile_and_fit(self, xy_train, xy_valid, bl_training_ratio, bl_validation_ratio, xy_noisy, xy_clean, gx_gy, indices, fit_args):
        ####### get and split the data
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
        x_noisy_train = x_noisy[:, indices_train, :]
        x_noisy_valid = x_noisy[:, indices_valid, :]
        print("x_noisy_train", x_noisy_train.shape, "x_noisy_valid", x_noisy_valid.shape)
        x_noisy_train_max_bounds = np.zeros((x_noisy_train.shape[1], x_noisy_train.shape[-1]))
        x_noisy_train_min_bounds = np.zeros((x_noisy_train.shape[1], x_noisy_train.shape[-1]))
        x_noisy_valid_max_bounds = np.zeros((x_noisy_valid.shape[1], x_noisy_valid.shape[-1]))
        x_noisy_valid_min_bounds = np.zeros((x_noisy_valid.shape[1], x_noisy_valid.shape[-1]))
    
        for noisy_feat in range(x_noisy_train.shape[-1]):
            x_noisy_train_max_bounds[:, noisy_feat], x_noisy_train_min_bounds[:, noisy_feat] = self.metric.extract_bounds(x_noisy_train[:, :, noisy_feat])
            x_noisy_valid_max_bounds[:, noisy_feat], x_noisy_valid_min_bounds[:, noisy_feat] = self.metric.extract_bounds(x_noisy_valid[:, :, noisy_feat])
        
        y_noisy_train = y_noisy[:, indices_train, ]
        
        x_clean_train = x_clean[indices_train, :]
        x_clean_valid = x_clean[indices_valid, :]
        y_clean_train = y_clean[indices_train,]
        y_clean_valid = y_clean[indices_valid,]
        ##########################################################################
        best_valid_loss = np.inf
        no_improvement = 0
        # TODO: read patience from fit_args
        patience = 10
        
        history = {'loss': []}
        training_constant = tf.constant(bl_training_ratio)
        valid_constant = tf.constant(bl_validation_ratio)
        input_features = xy_train[0].shape[1]       
        xy_train = tf.data.Dataset.from_tensor_slices(xy_train).batch(xy_train[0].shape[0])
        xy_valid = tf.data.Dataset.from_tensor_slices(xy_valid).batch(xy_valid[0].shape[0])
        history['val_loss'] = []
        
        self.compile(optimizer=self.optimizer, loss=self.loss_function)
        
        for epoch in range(fit_args_copy['epochs']):
            # Training loop
            for batch in xy_train:
                inputs, targets = batch
                with tf.GradientTape() as tape:
                    predictions = self(inputs)
                    loss = self.loss_function(targets, predictions) + penalty
                gradients = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                outer_dists = ["Euclidean"]
                weights = [1/input_features] * input_features
                ## TODO: double-check the following line reshape: expected shape=(None, 2), found shape=(2, 8, 2), so we need to reshape the data
                x_noisy_train = x_noisy_train.reshape(-1, x_noisy_train.shape[-1])
                y_noisy_train = self(x_noisy_train)
                training_ratio = self.metric.calculate_metric(x=inputs, y=targets,
                                                     x_bounds=(x_noisy_train_max_bounds, x_noisy_train_min_bounds), y_hat=y_noisy_train,
                                                     outer_dist=outer_dists, weights=weights, save=False, vis=False)

                training_ratio = tf.constant(training_ratio[outer_dists[0]]["Ratio"]) 
                training_ratio = tf.cast(training_ratio, tf.float32)
                penalty = tf.maximum(0., training_constant - training_ratio)                
                # Update the metrics
                self.train_loss.update_state(targets, predictions)                        
            # Validation loop
            for valid_batch in xy_valid:
                valid_inputs, valid_targets = valid_batch
                valid_predictions = self(valid_inputs)

                x_noisy_valid = x_noisy_valid.reshape(-1, x_noisy_valid.shape[-1])

                valid_y_noisy = self(x_noisy_valid)
                valid_ratio = self.metric.calculate_metric(x=x_clean_valid, y=y_clean_valid,
                                                           x_bounds=(x_noisy_valid_max_bounds, x_noisy_valid_min_bounds), y_hat=valid_y_noisy.numpy(),
                                                           outer_dist=outer_dists, weights=weights, save=False, vis=False)
                print("valid ratio", valid_ratio)
                valid_ratio = tf.constant(valid_ratio[outer_dists[0]]["Ratio"])
                valid_ratio = tf.cast(valid_ratio, tf.float32)
                valid_penalty = tf.maximum(0., valid_constant - valid_ratio)
                
                valid_loss = self.loss_function(valid_targets, valid_predictions) + valid_penalty
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    no_improvement = 0
                else:
                    no_improvement += 1
                self.valid_loss.update_state(valid_targets, valid_predictions)
            # Display the metrics at the end of each epoch
            template = 'Epoch {}, Loss: {}, Valid Loss: {}, Training Ratio: {}, Valid Ratio: {}'
            print(template.format(epoch+1,
                                  self.train_loss.result(),
                                  self.valid_loss.result(),
                                    training_ratio,
                                    valid_ratio))
            # Store the metrics in the history
            history['loss'].append(self.train_loss.result().numpy())
            # Reset the metrics for the next epoch
            self.train_loss.reset_states()
            self.valid_loss.reset_states()
            
            # Check for early stopping
            if no_improvement >= patience:
                print(f'Early stopping after {epoch} epochs')
                break
        return self, history
    def save_model(self, path):
        tf.keras.models.save_model(self, path)
    
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
        
    def load_model(self, path):
        model = tf.keras.models.load_model(path)
        return model