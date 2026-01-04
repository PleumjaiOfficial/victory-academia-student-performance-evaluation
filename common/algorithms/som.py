"""
# HR enhancement by AI

People Analysis - Y. Thornthan 

Date : 25/01/2024

**METHOD: SOM (Self-Organizing Map clustering)**

Bief:
1. Imagine a grid of neurons: Think of it like a honeycomb, where each hexagon (neuron) represents a group of data points.
2. Data points dance around: Each data point is fed into the SOM, and its "distance" to each neuron is calculated.
3. Winning neuron stretches: The closest neuron becomes the "winner," and it adjusts itself and its neighbors to be even closer to the data point.
4. The dance continues: As more data points are fed in, the winning neurons and their neighbors keep adjusting, forming clusters of similar data points on the map.

Benefits: 
1. Works well with high-dimensional data: Unlike some clustering methods that struggle with complex data, SOMs can handle it gracefully.
2. Reveals hidden relationships: By visualizing the data on a map, you can see how different groups are related to each other, uncovering patterns you might have missed otherwise.
3. Reduces dimensionality: The map itself is a lower-dimensional representation of the original data, making it easier to understand and interpret.

"""
import time
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output

def plot_SOM(X, mu, xy):

    #1D
    if X.shape[1] == 1:
         # Set the size of the figure
        plt.figure(figsize=(16, 8))
        plt.plot(X, np.zeros(len(X)), '.')
        plt.plot(mu, np.zeros(len(mu), 'or', mfc='none'))
    
    #time.sleep(1) # keep refresh rate of 0.25 seconds
    
    #2D
    if X.shape[1] == 2:
         # Set the size of the figure
        plt.figure(figsize=(16, 8))
        plt.plot(X[:, 0], X[:, 1], '.')
        plt.plot(mu[:, 0], mu[:, 1], 'or', mfc='none')
        for i, x in enumerate(xy):
            for j, y in enumerate(xy):
                if np.sum(np.abs(x - y)) == 1:
                    plt.plot(mu[[i, j], 0], mu[[i, j], 1], 'g')
    #time.sleep(1) # keep refresh rate of 0.25 seconds
    
    #3D
    if X.shape[1] > 2:
        # # Run PCA on the data and reduce the dimensions to pca_num_components dimensions
        # pca_num_components = 3
        # reduced_data = PCA(n_components=pca_num_components).fit_transform(X)
        # X = pd.DataFrame(reduced_data, columns=['pca1', 'pca2', 'pca3'])
        # X = X.values
        # print(X[:, 0])
        # plt.plot(X[:, 0], X[:, 1], '.')
        # plt.plot(mu[:, 0], mu[:, 1], 'or', mfc='none')

        #ax = Axes3D(plt.gcf())
        # Create a figure and a 3D axis
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(X[:, 0], X[:, 1], X[:, 2], '.')
        ax.plot(mu[:, 0], mu[:, 1], mu[:, 2], 'or', mfc='none')

        for i, x in enumerate(xy):
            for j, y in enumerate(xy):
                if np.sum(np.abs(x - y)) == 1:
                    plt.plot(mu[[i, j], 0], mu[[i, j], 1], mu[[i, j], 2], 'g')

        #time.sleep(1) # keep refresh rate of 0.25 seconds

        
        
        

def SOM(X, K, random_state = None,
        decay = 'linear',
        LR_0 = 0.5,
        LR_T= 0.1,
        Sigma_0 = 3,
        Sigma_T = 0.1,
        n_iter = 3e3,
        show = False,
        return_xy = False):
    
    """
    Implements a Self-Organizing Map (SOM) algorithm.

    Args:
        X (numpy.ndarray): Input data to cluster.
        K (int or tuple): Desired grid dimensions (e.g., 10 for 1D, (10, 10) for 2D).
        decay (str, optional): Decay type for learning rate and neighborhood radius. Defaults to 'linear'.
        LR_0 (float, optional): Initial learning rate. Defaults to 0.5.
        LR_T (float, optional): Final learning rate. Defaults to 0.1.
        Sigma_0 (float, optional): Initial neighborhood radius. Defaults to 3.
        Sigma_T (float, optional): Final neighborhood radius. Defaults to 0.1.
        n_iter (int, optional): Number of training iterations. Defaults to 1000.
        show (bool, optional): Visualize training progress. Defaults to False.
        return_xy (bool, optional): Return grid coordinates. Defaults to False.

    Returns:
        tuple: (idx, mu, xy) or (idx, mu)
            idx (numpy.ndarray): Indices of winning neurons for each data point.
            mu (numpy.ndarray): Final weights of the SOM neurons.
            xy (numpy.ndarray, optional): Grid coordinates of the neurons (if return_xy is True).
    """
    if random_state is not None:
        np.random.seed(random_state)  # Set the random seed
    
    if isinstance(K, int):
        K = [K]
        #decay = 'linear'
        #decay = 'nonlinear'
    
    n, d = X.shape

    # intital SOM
    if decay == 'nonlinear':
        radius = np.max(K)/2
        timeScale = n_iter / np.log(radius)

    # over dimension
    if len(K) > d:
        K = K[:d]
    
    N_node = np.prod(K)

    if len(K) == 1:
        # 1D grid
        mu = X[np.random.choice(len(X), K[0], replace=False)]
        mu = np.sort(mu)

        # grid coordinates
        xy = np.hstack((np.arange(N_node).reshape(N_node, 1), 
                        np.zeros((N_node, 1))))
        
        #print(mu)
        
    elif len(K) > 1:
        # 2D grid and more over
        # unravel_index : function converts a flat index or array of flat indices into a tuple of coordinate arrays.
        # REF -> https://betterprogramming.pub/the-numpy-illustrated-library-7531a7c43ffb

        """
            Syntax : numpy.unravel_index(indices, shape, order = ‘C’)
            Parameters :
            indices : [array_like] An integer array whose elements are indices into the flattened version of an array of dimensions shape.
            shape : [tuple of ints] "The shape" of the array to use for unraveling indices.
            order : [{‘C’, ‘F’}, optional] Determines whether the multi-index should be viewed as indexing in row-major (C-style) or column-major (Fortran-style) order.

            Return : [tuple of ndarray] Each array in the tuple has the same shape as the indices array.
        """

        xy = np.unravel_index(np.arange(N_node), K)
        xy = np.vstack(xy).T
        mu = xy.copy()

        # Keep dimension
        # weight
        mu = np.hstack((mu, np.random.rand(N_node, d-len(K))))    
        #print(mu)
  

    for t in range(int(n_iter)):
        # display iteration 
        # print("ITERATION : ", t)
        # print("\n")

        # select a sample
        # random the rample each round for test 
        v = X[np.random.choice(len(X))]

        # calculate distances
        D = np.sum((mu - v) ** 2, axis = 1)

        # 'BEST MATCHING UNIT'
        ## closest from Eucidian distant
        BMU = np.argmin(D)

        if decay == 'linear':
            Sigma2 = (Sigma_0 + (t/n_iter) * (Sigma_T -Sigma_0)) ** 2
            LR = LR_0 + (t/n_iter) * (LR_T - LR_0)

        if decay == 'nonlinear':
            Sigma2 = (radius * np.exp(-t / timeScale)) ** 2
            LR = LR_T * np.exp(-t / n_iter)

        # UPDATE WEIGHTED
        # Eucidian distant calculation
        # print("MATCHING UNIT: \n", xy)
        # print("\n")
        # print("BEST MATCHING UNIT: \n", xy[BMU])
        # print("\n")
        distBMU = np.sum((xy - xy[BMU]) ** 2, axis=1)

        # Neighbourhood function
        # For each unit in the Neighbourhood function of the WINNING neuron (BMU) update the weight 
        # the Neighbourhood function like expotiential(slope) mean the similary of data will capture in go down like gaussian 
        NB = np.exp(-distBMU / (2 * Sigma2)) 
        mu += (LR * NB * (v - mu).T).T # use transpose for bbroadcast trick

        # print("\nCalculation")
        # print("w(t + 1) = w(t) + Neighbourhood function(input - w(t) )")
        # print(mu, " + ", LR , "*", NB, "( ", v , " - ", mu ," )")
        # print("\n")
        

        
        # print matching unit
        #print(mu.shape)

        #print('%d: LR = %f BMU = %d' % (t, LR, BMU))

        if show:
            print("\n")
            g_0 = float(distBMU[0])
            g_0 = round(g_0, 4)
            print("distance group 0: ",g_0)

            g_1 = float(distBMU[1])
            g_1 = round(g_1, 4)
            print("distance group 1: ",g_1)

            g_2 = float(distBMU[2])
            g_2 = round(g_2, 4)
            print("distance group 2: ",g_2)


            print('%d: LR = %f BMU = %d' % (t, LR, BMU))
            print("\nweight: \n", mu)
            plt.clf()
            plot_SOM(X, mu, xy)
            plt.pause(1e-3)
            time.sleep(0.1) # keep refresh rate of 0.25 seconds
            clear_output(wait=True)
        
    # Distances
    K = np.prod(K)
    D = np.zeros((K, len(X)))
    for K in range(K):
        D[K] = np.sum((X - mu[K]) ** 2, axis=1)
    idx = np.argmin(D, axis = 0)    
    if return_xy:
        return idx, mu, xy
    
    else:
        return idx, mu