
import tensorflow as tf

class CustomLoss(tf.keras.losses.Loss):
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    def __init__(self, model, metric, y_clean, x_noisy, len_input_features, bl_ratio, nominator, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.metric = metric
        self.x_noisy = x_noisy
        self.input_features = len_input_features
        self.bl_ratio = bl_ratio
        self.nominator = nominator
        self.y_clean = y_clean
    
    
    def extract_g(self, y_noisy_tf, y_clean):
        y_noisy_min = tf.reduce_min(y_noisy_tf, axis=0)
        y_noisy_max = tf.reduce_max(y_noisy_tf, axis=0)
        y_noisy_min = tf.cast(y_noisy_min, tf.float64)
        y_noisy_max = tf.cast(y_noisy_max, tf.float64)
        #### use L2 as the inner distance between y and the min/max bounds
        y_dist_min = tf.sqrt(tf.square(y_noisy_min - y_clean))
        y_dist_max = tf.sqrt(tf.square(y_noisy_max - y_clean))

        gy = tf.maximum(y_dist_min, y_dist_max)
        
        mask_min = tf.equal(y_dist_min, gy)
        mask_max = tf.equal(y_dist_max, gy)
        # x_noisy_min_exp = tf.expand_dims(y_noisy_min, axis=0)
        # x_noisy_max_exp = tf.expand_dims(y_noisy_max, axis=0)
        g_points_min = tf.where(mask_min, y_noisy_min, tf.float32.min)
        g_points_max = tf.where(mask_max, y_noisy_max, tf.float32.min)
        g_points = tf.maximum(g_points_min, g_points_max)
        
        return g_points
    def apply_model(self, x):
        x_reshaped = tf.reshape(x, [-1, self.input_features])
        return self.model(x_reshaped)

    def call(self, y_true, y_pred):
        # Calculate the base loss (e.g., MSE loss)
        base_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        
        y_noisy = tf.vectorized_map(self.apply_model, self.x_noisy)
        y_noisy_reshaped = tf.reshape(y_noisy, (self.x_noisy.shape[0], self.x_noisy.shape[1], 1))
        gy = self.extract_g(y_noisy_reshaped, self.y_clean)
        gy_y_dist = tf.sqrt(tf.reduce_sum(tf.square(tf.cast(gy, tf.float64) - self.y_clean)))
        
        ratio = tf.math.divide(gy_y_dist, self.bl_ratio)
        ratio = tf.cast(ratio, tf.float64)
        base_loss = tf.cast(base_loss, tf.float64)
        loss = tf.math.add(base_loss, ratio)
        # # tf.print("base_loss: ", base_loss, "loss: ", loss, "gy_y_dist: ", gy_y_dist, "ratio: ", ratio)
        return loss

## class for the validation loss using the black formula, and inheriting from the CustomLoss class
class CustomLossBlack(CustomLoss):
    def __init__(self, model, metric, y_clean, x_noisy, len_input_features, bl_ratio, nominator, **kwargs):
        super().__init__(model, metric, y_clean, x_noisy, len_input_features, bl_ratio, nominator, **kwargs)
    
    def call(self, y_true, y_pred):
        base_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        y_noisy = tf.vectorized_map(self.apply_model, self.x_noisy)
        y_noisy_reshaped = tf.reshape(y_noisy, (self.x_noisy.shape[0], self.x_noisy.shape[1], 1))
        gy = self.extract_g(y_noisy_reshaped, self.y_clean)
        gy_y_dist = tf.sqrt(tf.reduce_sum(tf.square(tf.cast(gy, tf.float64) - self.y_clean)))
        ratio = tf.math.divide(gy_y_dist, self.bl_ratio)
        ratio = tf.cast(ratio, tf.float64)
        ratio = tf.math.add(tf.cast(tf.constant(1.), tf.float64), ratio)
        ratio = tf.cast(ratio, tf.float32)
        base_loss = tf.cast(base_loss, tf.float32)
        loss = tf.math.multiply(base_loss, ratio)
        return loss
        
class CustomMetric(tf.keras.metrics.Metric):
    def __init__(self, model, y_clean, x_noisy, len_input_features, nominator, **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.x_noisy = x_noisy
        self.input_features = len_input_features
        self.nominator = nominator
        self.y_clean = y_clean
        self.ratio = self.add_weight(name="ratio", initializer="zeros")

    def extract_g(self, y_noisy_tf, y_clean):
        y_noisy_min = tf.reduce_min(y_noisy_tf, axis=0)
        y_noisy_max = tf.reduce_max(y_noisy_tf, axis=0)
        y_noisy_min = tf.cast(y_noisy_min, tf.float64)
        y_noisy_max = tf.cast(y_noisy_max, tf.float64)
        y_dist_min = tf.sqrt(tf.square(y_noisy_min - y_clean))
        y_dist_max = tf.sqrt(tf.square(y_noisy_max - y_clean))

        gy = tf.maximum(y_dist_min, y_dist_max)
        
        mask_min = tf.equal(y_dist_min, gy)
        mask_max = tf.equal(y_dist_max, gy)
        g_points_min = tf.where(mask_min, y_noisy_min, tf.float32.min)
        g_points_max = tf.where(mask_max, y_noisy_max, tf.float32.min)
        g_points = tf.maximum(g_points_min, g_points_max)
        
        return g_points

    def apply_model(self, x):
        x_reshaped = tf.reshape(x, [-1, self.input_features])
        return self.model(x_reshaped)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_noisy = tf.vectorized_map(self.apply_model, self.x_noisy)
        y_noisy_reshaped = tf.reshape(y_noisy, (self.x_noisy.shape[0], self.x_noisy.shape[1], 1))
        gy = self.extract_g(y_noisy_reshaped, self.y_clean)
        gy_y_dist = tf.sqrt(tf.reduce_sum(tf.square(tf.cast(gy, tf.float64) - self.y_clean)))
        ratio = tf.math.divide(self.nominator, gy_y_dist)
        ratio = tf.reduce_mean(ratio)
        ratio = tf.cast(ratio, tf.float32)
        self.ratio.assign(ratio)

    def result(self):
        return self.ratio

    def reset_state(self):
        self.ratio.assign(0.)
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'model': self.model, 'y_clean': self.y_clean, 'x_noisy': self.x_noisy, 'len_input_features': self.input_features, 'nominator': self.nominator}

    @classmethod
    def from_config(cls, config):
        return cls(**config)