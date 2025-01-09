import numpy as np
import matplotlib.pyplot as plt
from Metric import RobustnessMetric
from scipy import integrate

def main():
    x = np.linspace(0, 10, 100)
    y = x**2

    # define a noise matrix N with dimensions 20 x 100
    N = np.random.normal(0, 0.1, (100, 100))

    # add noise to the x values like X_bar = x + N
    X_bar = x + N
    Y_bar = X_bar**2
    y_hat = np.mean(Y_bar, axis=0)
   
    # compute the barry center of X_bar using DTW
    x_barycenter_dtw = bc_dtw(x, X_bar)
    y_barycenter_dtw = bc_dtw(y_hat, Y_bar)
    # y_barycenter_dtw = y_hat
    
   
    # extract the bounds of X_bar, and name them u and l
    u_x = np.max(X_bar, axis=0)
    l_x = np.min(X_bar, axis=0)
    
    # bounds of Y_bar
    u_y = np.max(Y_bar, axis=0)
    l_y = np.min(Y_bar, axis=0)
    

    metric = RobustnessMetric.RobustnessMetric()
    
    gx = metric.extract_g(x, x_bounds=(l_x, u_x))
    gy = metric.extract_g(y_hat, x_bounds=(l_y, u_y))
    
    x_bar_u, x_bar_l = metric.extract_bounds(X_bar)
    y_bar_u, y_bar_l = metric.extract_bounds(Y_bar)
    
    
    gx_y_barycenter = np.vstack((gx, y_barycenter_dtw))
    
    # sort gx_y_barycenter using the first column
    index_mapping = np.argsort(gx_y_barycenter[0])
    inverse_inex_mapping = np.argsort(index_mapping)
    # gx_y_barycenter = gx_y_barycenter[:, index_mapping]
    u_x, l_x = bound_curves(gx_y_barycenter[1], gx_y_barycenter[0])
    u_y, l_y = bound_curves(gx_y_barycenter[0], gy)
    # Assuming u_x and l_x are numpy arrays and are defined over the same x range
    area = integrate.trapz(np.abs(u_x - l_x))

    print("The area between the two lines is:", area)
    # Assuming gx_y_barycenter[1,:] is a numpy array and is defined over the same x range as u_x and l_x
    output_area = integrate.trapz(np.abs(gx_y_barycenter[1,:][np.argsort(u_x)] - gx_y_barycenter[1,:][np.argsort(l_x)]))
    # now caclulate the area between the two lines on the output side
    y_area = integrate.trapz(np.abs(u_y - l_y))
    
    print("The area between the two lines on the output side is:", output_area)
    print("The area between the two lines on the output side is:", y_area)
    print("the area of x_bar_u and x_bar_l is: ", integrate.trapz(np.abs(x_bar_u - x_bar_l)))
    print("the area of y_bar_u and y_bar_l is: ", integrate.trapz(np.abs(y_bar_u - y_bar_l)))
    print("ratio of the area of x_bar_u and x_bar_l to the area of y_bar_u and y_bar_l is: ", integrate.trapz(np.abs(x_bar_u - x_bar_l))/integrate.trapz(np.abs(y_bar_u - y_bar_l)))
    print(" ratio if input to output is: ", area/output_area)
    # inverse the u_y and l_y to match the original indices of gx_y_barycenter
    # u_y = u_y[inverse_inex_mapping]
    # l_y = l_y[inverse_inex_mapping]
    # plt.plot(x,y, label='$x$ vs. $y$')
    # plt.plot(gx, gy, label='$G(x, y)$')
    # plt.legend()
    # plt.show()
    # sort gx 
    # gx_sorted_indices = np.argsort(gx)
    # gx_sorted = gx[gx_sorted_indices]
    
    # gy_sorted = gy[gx_sorted_indices]
    # plt.plot(x[gx_sorted_indices], y[gx_sorted_indices], label='$x$')
    # # plt.plot(gx, gy, label='$G(x, \hat{X})$ vs. $G(y, \hat{Y})$')
    # plt.plot(gx_sorted, gy_sorted, label='$G(x, \hat{X})$ vs. $G(y, \hat{Y})$ - sorted')
    # plt.legend()
    # plt.show()
    # bounds of X_bar
    plt.plot(x, x, label='$x$')
    plt.plot(x, gx, label='$G(\hat{X}, x)$')
    plt.plot(x, x_bar_u, label='$\hat{X}_u$')
    plt.plot(x, x_bar_l, label='$\hat{X}_l$')
    # plt.plot(x, u_x, label='$u_x$', linestyle='--')
    # plt.plot(x, l_x, label='$l_x$', linestyle='--')
    
    plt.legend()
    plt.show()
    
    # # bounds of Y_bar
    # plt.plot(x, y, label='$x$ vs. $y$')
    # plt.plot(x, gy, label='$G(x, y)$')
    # plt.plot(x, y_bar_u, label='$\hat{Y}_u$')
    # plt.plot(x, y_bar_l, label='$\hat{Y}_l$')
    # plt.legend()
    # plt.show()
    
    plt.plot(x, x, label='$x$')
    plt.plot(x, gx, label='$G(x)$')
    plt.plot(x, u_x, label='$u_x$')
    plt.plot(x, l_x, label='$l_x$')
    plt.legend()
    plt.show()
    
    
    plt.figure()
    plt.plot(x, y, label='$x$ vs. $y$')
    plt.plot(gx, y_barycenter_dtw, label='$G(\hat{X})$ vs. $y$', linestyle='--')
    # plt.plot(l_x, y_barycenter_dtw, label='$BC(\hat{Y})$ vs. $l(\hat{X})$', linestyle='--')
    plt.legend()
    plt.show()
    
    # find the point-wise drift between barrycenter of x_hat, and the bounds u_x and l_x
    drift_u = np.max(np.abs(x_barycenter_dtw - u_x))
    drift_l = np.max(np.abs(x_barycenter_dtw - l_x))
    
    print(f"Drift between barrycenter of x_hat and u_x: {drift_u}")
    print(f"Drift between barrycenter of x_hat and l_x: {drift_l}")

    # plot the bounds
    plt.figure()
    plt.plot(x, x, label='$x$')
    # plt.plot(x, u_x, label='$u(\hat{X})$')
    plt.plot(x, x_barycenter_dtw + drift_u, label='$BC(\hat{X})$', linestyle='--')
    plt.plot(x, x_barycenter_dtw - drift_l, label='$BC(\hat{X})$', linestyle='--')
    # plt.plot(x, l_x, label='$l(\hat{X})$')
    plt.plot(x, x_barycenter_dtw, label='$BC(\hat{X})$', linestyle='--')
    plt.legend()
    

    plt.figure()
    # plot the bounds
    plt.plot(x, y, label='$y$')
    plt.plot(x, y_hat, label='$\hat{y}$')
    # plt.plot(x, gy, label='$G(y, \hat{y})$')
    # plt.plot(gx, gy, label='$G(y, \hat{y})$', color='red')
    # plt.plot(x, gx**2, label='$f(G(x, \hat{X}))$')
    # plt.plot(x, u_y, label='$u(\hat{Y})$')
    # plt.plot(x, l_y, label='$l(\hat{Y})$')
    # plt.plot(u_x, y_barycenter_dtw, label='$BC(\hat{Y})$ vs. $u(\hat{X})$', linestyle='--')
    # plt.plot(l_x, y_barycenter_dtw, label='$BC(\hat{Y})$ vs. $l(\hat{X})$', linestyle='--')
    # plt.plot(gx, y_barycenter_dtw, label='$BC(\hat{Y})$ vs. $G(x, \hat{X})$', linestyle='--')
    # plt.plot(gx_y_barycenter[0,:], gx_y_barycenter[1,:], label='$BC(\hat{Y})$ vs. $G(x, \hat{X})$', linestyle='--', color='purple')
    plt.plot(gx_y_barycenter[0,:], gx_y_barycenter[1,:], label='$\hat{y}$ vs. $G(x, \hat{X})$', linestyle='--', color='purple')
    plt.plot(u_x, gx_y_barycenter[1,:], label='$u_{y}$', color='green')
    plt.plot(l_x, gx_y_barycenter[1,:], label='$l_{y}$', color='red')
    plt.legend()

    plt.show()

def bound_curves(x, y):
    # given the curve y = f(x), compute the upper bound curve
    y_upper = np.zeros(y.shape)
    y_lower = np.zeros(y.shape)
    for i in range(1, 100):
        y_upper[i] = np.max(y[:i+1])
        y_lower[i] = np.min(y[i:])
    
    return y_upper, y_lower

def bc_dtw(x, X_bar):
    # compute the distance matrix
    D = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            D[i, j] = (X_bar[i] - X_bar[j]).T @ (X_bar[i] - X_bar[j])
    # compute the cumulative distance matrix
    C = np.zeros((100, 100))
    C[0, 0] = D[0, 0]
    for i in range(1, 100):
        C[i, 0] = C[i-1, 0] + D[i, 0]
        C[0, i] = C[0, i-1] + D[0, i]
    for i in range(1, 100):
        for j in range(1, 100):
            C[i, j] = D[i, j] + min(C[i-1, j], C[i, j-1], C[i-1, j-1])
    # compute the barry center
    barycenter_dtw = np.zeros(100)
    i = 99
    j = 99
    while i > 0 and j > 0:
        barycenter_dtw[i] = x[i]
        if C[i-1, j] < C[i, j-1] and C[i-1, j] < C[i-1, j-1]:
            i -= 1
        elif C[i, j-1] < C[i-1, j] and C[i, j-1] < C[i-1, j-1]:
            j -= 1
        else:
            i -= 1
            j -= 1
    barycenter_dtw[0] = x[0]
    return barycenter_dtw

if __name__ == '__main__':
    main()
