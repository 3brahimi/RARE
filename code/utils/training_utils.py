
import tensorflow as tf

class CustomLoss(tf.keras.losses.Loss):
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    def __init__(self, model, metric, y_clean, x_noisy, len_input_features, bl_ratio, nominator, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.metric = metric
        self.x_noisy = x_noisy
        self.input_features = len_input_features
        self.bl_ratio = bl_ratio
        self.nominator = nominator
        self.y_clean = y_clean
    
    
    def extract_g(self, y_noisy_tf, x_clean):
        y_noisy_min = tf.reduce_min(y_noisy_tf, axis=0)
        y_noisy_max = tf.reduce_max(y_noisy_tf, axis=0)
        y_dist_min = tf.abs(y_noisy_min - y_noisy_tf)
        y_dist_max = tf.abs(y_noisy_max - y_noisy_tf)
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
        # x_noisy_reshaped = tf.reshape(self.x_noisy, [-1, self.input_features])
        batch_size = 32  # Define your batch size

        y_noisy = tf.vectorized_map(self.apply_model, self.x_noisy)
        # y_noisy = self.model(x_noisy_reshaped)
        y_noisy_reshaped = tf.reshape(y_noisy, (self.x_noisy.shape[0], self.x_noisy.shape[1], 1))
        gy = self.extract_g(y_noisy_reshaped, y_true)
        gy_y_dist = tf.sqrt(tf.reduce_sum(tf.square(gy -self.y_clean)))
        
        # ratio =  (gy_y_dist / self.nominator) + 1
        ratio = self.nominator / gy_y_dist
        # make sure both are of the same type of float64
        ratio = tf.cast(ratio, tf.float16)
        diff_ratio = tf.math.subtract(self.bl_ratio, ratio)

        diff_ratio = tf.math.subtract(self.bl_ratio, ratio)
        min_bound = tf.constant(0.)
        min_bound = tf.cast(min_bound, tf.float16)
        penalty = tf.reduce_mean(tf.maximum(min_bound, diff_ratio))

        # Return the total loss
        # return base_loss * ratio
        # add the penalty to the base loss
        base_loss = tf.cast(base_loss, tf.float16)
        # penalty = tf.cast(tf.constant(0.), tf.float64)
        loss = tf.math.add(base_loss, penalty)
        # tf.print("base_loss: ", base_loss, "penalty: ", penalty, "loss: ", loss)
        return loss