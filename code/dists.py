from matplotlib.pylab import figure
import matplotlib.pylab as plt
import os
import similaritymeasures as sm
from scipy.spatial.distance import directed_hausdorff
import numpy as np

### calculate distance between each point and the max, each point and the min
def calculate_ditances(x, x_max, x_min):
    distsmax, distsmin = [], []
    distsmax = abs(x_max - x)
    distsmin = abs(x_min - x)
    # for idx, item in enumerate(x):
        
    #   ### L1 distance between each point and the max
    #   distsmax.append(abs(x_max[idx] -  item))
      
    #   ### L1 distance between each point and the min
    #   distsmin.append(abs(x_min[idx] -  item))
    return [distsmax, distsmin]


## calculate cosine distance between each point and the max, each point and the min
def calculate_cosine_distance(x, y, domain="waveform"):
    # calculate the cosine similarity
    if len(x) != len(y):
    # pad the shorter vector with zeros
        if len(x) < len(y):
            x = np.concatenate((x, np.zeros(len(y) - len(x))))
        else:
            y = np.concatenate((y, np.zeros(len(x) - len(y))))
    if domain =="waveform":
        cs = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        return (1 - cs)/ 2
    else:
        from sklearn.metrics.pairwise import cosine_distances
        cs = cosine_distances(x, y)
        print("cs", np.mean(cs))
        return np.mean(cs).tolist()
def dtw(x, y):
    dist, path = dtw(x.T, y.T)
    dtw_distance = dist / len(path)
    # return sm.dtw(x, y, keep_internals=True, distance_only=True)
    return dtw_distance

## calculate the point-wise cosine distance between each point and the max, each point and the min
def point_wise_cosine_distance(x, x_max, x_min):
    distsmax, distsmin = [], []
    from scipy.spatial.distance import cosine 

    for idx, item in enumerate(x):
        distsmax.append(1 - cosine(np.array(x_max[idx]), np.array(item)))
        distsmin.append(1 - cosine(np.array(x_min[idx]), np.array(item)))
    
    return [distsmax, distsmin]

## plot distances between min and max at each point as vertical lines
def plot_distances(x, x_min, x_max, vis=False, save=True, fig_name="Distances plot", path="./" ):
    
    figure(figsize=(8,8))
    
    for idx, point in enumerate(x):
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
def plot_max_distance(x, distances_array, dist_min, dist_max, x_min, x_max, vis=False, save=True, fig_name="Max distances", path="./"):
    plt.clf()
    fig, ax = plt.subplots(figsize=(6, 6))

    distsminmax_y = []
    maxaxs = []
    handelA, handelB = None, None
    for idx, point in enumerate(x):
        if distances_array[idx] == dist_min[idx]:
            distsminmax_y.append(x_min[idx])
        # handelA = plt.vlines(point, ymin=x_min[idx], ymax = x_min[idx]+dist_min[idx], colors='g')
            handelA = plt.vlines(point, ymin=x_min[idx], ymax = x_min[idx]+dist_min[idx], colors='g', alpha=0.1)
        else:
            distsminmax_y.append(x_max[idx])
            handelB =plt.vlines(point, ymin=x_max[idx]-dist_max[idx], ymax =x_max[idx] , colors='r', alpha=0.1)
        
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


def plot_max_distance_curves(x, true_sloution, distances_array, dist_min, dist_max, x_min, x_max, vis=False, save=True, fig_name="Max distances curve", path="./"):
    plt.clf()
    fig, ax = plt.subplots(figsize=(6, 6))

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


def min_max_bounds(df):
    
    min_bound = np.array(df.min(axis=1))
    max_bound = np.array(df.max(axis=1))
    
    return min_bound, max_bound


# Euclidean distance
def L2_distance(x, y, type="pointwise"):
    if type == "pointwise":
        return [np.sqrt((a-b)*(a-b)) for a, b in zip(x, y)]
    else:
        return np.sqrt(np.sum([(a-b)*(a-b) for a, b in zip(x, y)]))

# Frechet distance
def frechet_distance(x, y):
    xaxis = np.arange(0, len(x))
    repart_x = np.column_stack((xaxis, x))
    repart_y = np.column_stack((xaxis, y))

    return sm.frechet_dist(repart_x, repart_y)

# Hausdorff distance
def hausdorff_distance(x, y):
    xaxis = np.arange(0, len(x))
    repart_x = np.column_stack((xaxis, x))
    repart_y = np.column_stack((xaxis, y))
    return directed_hausdorff(repart_x, repart_y)[0]
    
def L1_distance(x, y, type="pointwise"):
    if type == "pointwise":
        return abs(x-y)
    else:
        return np.sum([abs(a-b) for a, b in zip(x, y)])