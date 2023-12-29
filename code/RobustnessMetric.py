import numpy as np
import matplotlib.pyplot as plt
import os 
import dists as calculations

class RobustnessMetric:
    def __init__(self):
        self.dists_functions = {
            "L1": calculations.L1_distance,
            "L2": calculations.L2_distance,
            "Euclidean": calculations.L2_distance,
            "Frechet": calculations.frechet_distance,
            "Hausdorff": calculations.hausdorff_distance,
        }
        
    def calculate_ditances(self, x, max_bound, min_bound, dist="L1"):
        distsmax = []
        distsmin = []
        if dist == "L1":
            dist_fn = self.dist_L1_norm
        
        elif dist =="L2":
            dist_fn = self.dist_L2_norm
        
        else:
            raise ValueError("Not supported distance function, Available options are only L1 and L2")
        
        distsmin = dist_fn(x,min_bound) 
        distsmax = dist_fn(x,max_bound) 
        return [distsmax, distsmin]
    
    def extract_bounds(self, x_noisy):
        """
        Extracts the max and min bounds of the noisy set.

        Parameters:
        - x_noisy: The modulated input data with noise.

        Returns:
        - Two signals with the maximum and minimum bounds of the noisy set.
        """
        max_bound = np.max(x_noisy, axis=0)
        min_bound = np.min(x_noisy, axis=0)
        # Return the two bounds
        return max_bound, min_bound
        
    # Create a function to aggregate the distances according to the aggregation function
    def aggregate_Q(self, distances, Q="max"):
        func_dict = {
            "max": np.max,
            "min": np.min,
            "mean": np.mean,
            "median": np.median
        }
        if Q in func_dict.keys():
            return func_dict[Q](distances, axis=0)
        else:
            raise ValueError("Not supported aggregation function, Available options are only max, min, mean and median")
    
    # Create a function that constructs G(x) from the aggregated distances
    def construct_G(self, aggregated_distances, distsmax, distsmin, min_bound, max_bound, Q):
        if Q == "max" or Q =="min":
            distsmax_mask = [1 if distsmax[i] == aggregated_distances[i] else 0 for i in range(len(aggregated_distances))]
            distsmin_mask = [1 if distsmin[i] == aggregated_distances[i] else 0 for i in range(len(aggregated_distances))]
            min_bound_aggregatd = np.multiply(min_bound, distsmin_mask)

            max_bound_aggregatd = np.multiply(max_bound, distsmax_mask)
            return {
                "G": min_bound_aggregatd + max_bound_aggregatd,
                "G_distances": np.multiply(aggregated_distances, distsmax_mask) + np.multiply(aggregated_distances, distsmin_mask),
                "G_min_bound": np.multiply(min_bound, distsmin_mask),
                "G_max_bound": np.multiply(max_bound, distsmax_mask)
                }
        
        elif Q == "mean" or Q == "median":
            return {
                "G": min_bound + aggregated_distances
                }
        else:
            raise ValueError("Not supported aggregation function, Available options are only max, min, mean and median")    



    ## plot distances between min and max at each point as vertical lines
    def plot_distances(self, t, x_min, x_max, vis=True, save=False, fig_name="Distances plot", path="./" ):
        """
        Plot distances between min and max at each point as vertical lines

        Args:
            x_min (Array): minimum bound
            y_min (Array): maximum bound
            vis (bool, optional): Show plot. Defaults to False.
            save (bool, optional): Save plot. Defaults to True.
            fig_name (str, optional): Figure name. Defaults to "RM".
            path (str, optional): Figure save path. Defaults to "./".
        """
        
        plt.figure(figsize=(8,8))
        print("x_minx_min", x_min)
        for idx, point in enumerate(t):
          plt.vlines(point, ymin=x_min[idx], ymax = x_max[idx])

        plt.xlabel("X", fontsize=18)
        plt.ylabel("Y", fontsize=18)
        plt.title("Differences between min and max")
        plt.legend(prop={'size': 14}, shadow=True, handlelength=1.5, fontsize=14)

        if save:
            if not os.path.isdir(path):
                os.makedirs(path)
            plt.savefig(f"{path}/{fig_name}.pdf")
            if vis:
                plt.show()
                plt.clf()
        if vis:
            plt.show()
            plt.clf()


    """
    plot max distance such that:
    for every xi: yi= max(dist(xi, max), d(xi,min))
    """  
    def plot_aggregated_distance(self, x, aggregated_distances, distsmin, distsmax, min_bound, max_bound, vis=True, save=False, fig_name="Max distances", path="./"):
        plt.figure(figsize=(6,6))

        distsminmax_y = []
        maxaxs = []
        handelA, handelB = None, None
        for idx, point in enumerate(x):
            if aggregated_distances[idx] == distsmin[idx]:
                distsminmax_y.append(min_bound[idx])
            # handelA = plt.vlines(point, ymin=x_min[idx], ymax = x_min[idx]+dist_min[idx], colors='g')
                handelA = plt.vlines(point, ymin=min_bound[idx], ymax = min_bound[idx]+distsmin[idx], colors='g', alpha=0.1)
            else:
                distsminmax_y.append(max_bound[idx])
                handelB =plt.vlines(point, ymin=max_bound[idx]-distsmax[idx], ymax =max_bound[idx] , colors='r', alpha=0.1)
            
        plt.xlabel("X", fontsize=18)
        plt.ylabel("Y", fontsize=18)

        plt.title("for every xi: yi = max(d(xi, max), d(xi,min))")
        plt.legend(prop={'size': 14}, handles=[handelA, handelB], labels=["min is greater", "max is greater"], fontsize=14)
        
        if save:
            if not os.path.isdir(path):
                os.makedirs(path)
            plt.savefig(f"{path}/{fig_name}.pdf",bbox_inches='tight')
        if vis:
            plt.show()    


    def plot_G(x, true_sloution, distances_array, dist_min, dist_max, x_min, x_max, vis=False, save=True, fig_name="Max distances curve", path="./"):
        plt.figure(figsize=(6,6))

        distsminmax_y = []
        maxaxs = []
        handelA, handelB = None, None
        for idx, point in enumerate(x):
            if distances_array[idx] == dist_min[idx]:
                distsminmax_y.append(x_min[idx])
                handelA = plt.vlines(point, ymin=x_min[idx], ymax = x_min[idx]+dist_min[idx], colors='g', alpha=0.3)
            else:
                distsminmax_y.append(x_max[idx])
                handelB =plt.vlines(point, ymin=x_max[idx]-dist_max[idx], ymax =x_max[idx] , colors='r', alpha=0.3)
            
        plt.xlabel("X", fontsize=18)
        plt.ylabel("Y", fontsize=18)

        plt.title("for every xi: yi = max(d(xi, max), d(xi,min))")

        handelC = plt.plot(x, distsminmax_y, label = "Max. distances curve", alpha =1)
        handelD = plt.plot(x, true_sloution, label = "True solution", alpha=1)
        plt.legend(prop={'size': 14}, shadow=True, fontsize=14)


        if save:
            if not os.path.isdir(path):
                os.makedirs(path)
            plt.savefig(f"{path}/{fig_name}.pdf", bbox_inches='tight')
            plt.close()
            
        if vis:
            plt.show()
        
        return distsminmax_y
    
    
    # Create G of the given input
    def create_G(x, dist, vis=False, save=True, fig_name="G", path="./"):
        plt.clf()
        fig, ax = plt.subplots(figsize=(6, 6))
        G = np.zeros((len(x), len(x)))
        for i in range(len(x)):
            for j in range(len(x)):
                G[i,j] = dist(x[i], x[j])
        
        plt.imshow(G, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title("G")
        plt.xlabel("X", fontsize=18)
        plt.ylabel("Y", fontsize=18)
        if save:
            if not os.path.isdir(path):
                os.makedirs(path)
            plt.savefig(f"{path}/{fig_name}.pdf", bbox_inches='tight')
        if vis:
            plt.show()
            plt.close()
        return G

    def calculate_dist(self, x, y, dist, **kwargs):
        if isinstance(dist, str):
                if dist in self.dists_functions:
                    dist_fn = self.dists_functions[dist]
                else:
                    raise ValueError("Not supported distance function. Available options are:", list(self.dists_functions.keys()))
        elif callable(dist):
            dist_fn = dist
        else:
            raise ValueError("Invalid type for dist. It should be either a string or a callable function.")
        return dist_fn(x, y, **kwargs)
    
    # return the ratio of the input to the output of the "d"
    def metric_ratio(self, true_input, g_input, g_output, true_output, dist, **kwargs):
        print("Calculating the ratio between the input and the output")
        # read type from kwargs
        type = kwargs.get("type")
        print("type is:", type)
        results = {}
        for d in dist:
            dx = self.calculate_dist(true_input, g_input, d, type=type)
            dy = self.calculate_dist(true_output, g_output, d, type=type)
            results[d] = {
            "Input distance": dx,
            "Output distance": dy,    
            "Ratio": dx/dy 
            }
        return results
        
    def bounds_distance(self, x, bounds, dist):
        print(dist)
        distsmin = self.calculate_dist(x, bounds[0], dist)
        distsmax = self.calculate_dist(x, bounds[1], dist)
        return [distsmin, distsmax]

    
    # Calculate the RM between the true input, noisy input and the true output, noisy output 
    def calculate_metric(self, x, y, x_bounds=None, y_bounds=None, x_hat=None, y_hat=None,
                         Q="max", inner_dist = "L2", outer_dist =["Euclidean"], vis=False, save=True, fig_name="RM", path="./"):
        """
        Calculate the RM between the true input, noisy input and the true output, noisy output

        Args:
            x (Array): True input
            y (Array): Ground Truth output
            x_bounds (tuple): Min and max bounds of the true input
            y_bounds (tuple): Min and max bounds of the true output
            x_hat (Array): Noisy input. Defaults to None.
            y_hat (Array): Noisy output. Defaults to None.
            vis (bool, optional): Show plot. Defaults to False.
            save (bool, optional): Save plot. Defaults to True.
            fig_name (str, optional): Figure name. Defaults to "RM".
            path (str, optional): Figure save path. Defaults to "./".
        """
        if x_hat is not None and y_hat is not None:
            x_bounds = self.extract_bounds(x_hat)
            y_bounds = self.extract_bounds(y_hat)
             
        # min_input_bound, max_input_bound = calculations.min_max_bounds(x_hat)
        x_dists = self.bounds_distance(x, (x_bounds[0], x_bounds[1]), inner_dist)
        
    
        x_dists_agg = self.aggregate_Q(x_dists, Q)

        y_dists = self.bounds_distance(y, (y_bounds[0], y_bounds[1]), inner_dist)
        
        y_dists_agg = self.aggregate_Q(y_dists, Q)
        
        # construct G of input 
        gx = self.construct_G(x_dists_agg, x_dists[0], x_dists[1], x_bounds[0], x_bounds[1], Q)
        
        
        # plot_signal_bounds(testX[:,0].flatten(), testY, test_predict, min_arr, max_arr,vis=False, save=True, fig_name=f"Signal bounds for output for noise {i}", path=f"{model_path}/{distribution}/{noise_type}/{variance_type}/results/signal_bounds_output")

        # self.plot_distances(t, min_input_bound, max_output_bound)
        # self.plot_aggregated_distance(t, distsmaxmin_input, input_dists[1], input_dists[0], min_input_bound, max_input_bound)
        # TODO: plot G of input
        # TODO: plot G of output
        # TODO: plot bounds of input
        # TODO: plot bounds of output
        
        # construct G of output
        gy = self.construct_G(y_dists_agg, y_dists[0], y_dists[1], y_bounds[0], y_bounds[1], Q)
        
        robustness = self.metric_ratio(x, gx["G"], y, gy["G"], outer_dist, type="overall")
        return robustness
    
    # extract_g function that takes the clean vector x, a noisy set x_hat or x_bounds and a distance function
    # it returns a signinificant noisy signal gx that is the most distant from the clean signal x
    def extract_g(self, x, x_hat=None, x_bounds=None, dist="L2", Q="max"):
        """
        Extract a significant noisy signal from the noisy set with a reference point x.

        Args:
            x (Array): Reference signal
            x_bounds (tuple): Min and max bounds of the noisy set. Defaults to None.
            x_hat (Array): Noisy input. Defaults to None.
            dist (str, optional): Distance function. Defaults to "L2".
            Q (str, optional): Aggregation function. Defaults to "max".
        """        
        if x_hat is not None:
            x_bounds = self.extract_bounds(x_hat)
            
        x_dists = self.bounds_distance(x, (x_bounds[0], x_bounds[1]), dist)

        x_dists_agg = self.aggregate_Q(x_dists, Q)
        # construct G of input 
        gx = self.construct_G(x_dists_agg, x_dists[0], x_dists[1], x_bounds[0], x_bounds[1], Q)
        
        return gx["G"]
    
    def add_dist(self, dist, func):
        self.dists_functions[dist] = func

# if __name__=="__main__":
#     metric = RobustnessMetric()
#     x = np.arange(0, 10, 0.1)
#     y = 2*x
#     x_noisy = []
#     for i in range(10):
#         x_noisy_i = x + np.random.normal(0, 0.9, len(x))
#         x_noisy.append(x_noisy_i)
#     # x_noisy = np.array([x_noisy_i for i in range(10)])
#     # x_noisy = x + np.random.normal(0, 0.9, len(x))
#     # replicate y 10 times
#     y_noisy = np.repeat(y[np.newaxis, :], 10, axis=0)
#     print(metric.calculate_metric(x, y, x_hat=x_noisy, y_hat=y_noisy, outer_dist=["Euclidean", "L1"]))
    