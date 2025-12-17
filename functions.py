import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture 
from sklearn.preprocessing import scale



# Interactive graphs of the pre/post data for each marker. 

def update_marker(marker_num, view, pre_data, post_data, viewing_options, ax, fig):
    """
    Updater function to allow the viewer to slide through the data for each marker.
    Shows the relationship between pre and post data, as well as allows viewer to see the distributions of the pre/post data separately. 

    Args:
        marker_num (int): Which marker the viewer wants to see. 
        view (str): Which data the viewer wants to see. 
        viewing_options (list(str)): All viewing options. (fixed)
        ax: axes to draw updates on (fixed)
        fig: figure to update (fixed)
    """
    
    # Creates arrays of the pre and post data for the required marker
    marker_i_pre = pre_data[f'marker_{marker_num}']
    marker_i_post = post_data[f'marker_{marker_num}']

    # Clears previous data. 
    ax.cla()
    
    # Scatter plot of pre data and post data
    if view==viewing_options[0]:
        ax.scatter(marker_i_pre, marker_i_post, alpha=0.2, color='deepskyblue')
        ax.set_xlabel('Pre data')
        ax.set_ylabel('Post data')   
       
    # Histogram to show distribution of pre data  
    elif view==viewing_options[1]:
        ax.hist(marker_i_pre, bins=150)
        ax.set_xlabel('Pre data')
        ax.set_ylabel('Frequency')
 
    # Histogram to show distribution of post data  
    elif view==viewing_options[2]:
        ax.hist(marker_i_post, bins=150, orientation='horizontal')
        ax.set_xlabel('Frequency')   
        ax.set_ylabel('Post data')         

    ax.set_title(f'Marker {marker_num}')
    ax.grid()
    ax.set_axisbelow(True)
    fig.canvas.draw()
    
    
    
def plot_markers(marker_x, marker_y, post_data, ax, fig):
    """
    Updater function to plot each marker against each other, with the correlation value plotted above the figure. 

    Args:
        marker_x (array): array of marker values to be plotted on x axis.
        marker_y (array): array of marker values to be plotted on y axis.
    """
    
    x_data = post_data[marker_x]
    y_data = post_data[marker_y]
    xy_merged = pd.concat([x_data, y_data], axis=1)
   
    ax.cla()
    
    # Scatter plot of the selected markers
    ax.scatter(x_data, y_data, alpha=0.7, color='deepskyblue')
    ax.set_xlabel(marker_x)
    ax.set_ylabel(marker_y)
    
    # Sets title
    fig.suptitle(f'{marker_x} against {marker_y}')
    
    # Prints correlation of the two markers above the figure
    # correlation = x_data.corr(y_data)
    correlation = xy_merged.corr()
    correlation = correlation.iloc[0,1]
    ax.set_title(f'Correlation: {correlation:.2f}')
    ax.title.set_size(8)
    plt.show()
    
    
def append_pc(data, n_components=6):
    """
    Adds principal component columns to dataframes

    Args:
        data (dataframe): The data that the PCA will be run on. 
        n_components (int, optional): Number of components in PCA analysis. Defaults to 6.

    Returns:
        data: updated dataframe with new columns added on. 
        pca: The PCA fitting for this data. 
    """
    
    # Fit PCA to data
    pca = PCA(n_components=n_components)
    pca = pca.fit(scale(data))
    
    # Data must be scaled so that magnitude of data does not skew analysis
    X_new = pca.fit_transform(scale(data))

    # Appends Principal Component values to the dataframes, with number of components used described by i. 
    for i in range(0, n_components):
        data[f'PC{i+1}'] = X_new[:, i]
        
    return data, pca



# Interactive plot to show pca analysis in 2d/3d

def vary_num_components(num_dims, pre_data, post_data, titles, ax, fig):
    """
    Updater function which plots the first principal components against each other.

    Args:
        num_dims (string): option to view the PCA data in 2D, 3D, or 1D. 
    """
   
    # List to iterate through both pre and plot to make plotting simpler  
    pre_post = [pre_data, post_data]
      
    # Initialises figure
    # Needs to done in function due to switching of projection into 3D
    fig = plt.figure(figsize=(10,4))
      
    if num_dims=='1d':
        for i, name in enumerate(pre_post):
            ax = fig.add_subplot(1, len(pre_post), i+1)
            ax.hist(name['PC1'], bins=100)

            ax.set_ylabel('Frequency')
            
       
    elif num_dims=='2d':
        for i, name in enumerate(pre_post):
            ax = fig.add_subplot(1, len(pre_post), i+1)
            ax.scatter(name['PC1'], name['PC2'], alpha=0.2, color='deepskyblue')
            
            # Patient zero
            ax.scatter(name['PC1'][0], name['PC2'][0], color='black', marker='*')

            ax.set_ylabel('PC2')

    else:
        for i,data in enumerate(pre_post):
            ax = fig.add_subplot(1, len(pre_post), i+1, projection='3d')
            ax.scatter(data['PC1'], data['PC2'], data['PC3'], alpha=0.2, color='deepskyblue')
            
            # Patient zero
            ax.scatter(data['PC1'][0], data['PC2'][0], data['PC3'][0], color='black')

            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
    
    ax.set_xlabel('PC1')        
    ax.set_title(titles[i])
    
    fig.suptitle(f'PCA analysis of both sets of data shown in {num_dims}') 
    fig.canvas.draw()
    
    plt.show()



# Function used to show clusters 

def plot_clusters2(num_dims, n_clusters, post_data):
    """
    Function that takes an array containing the pca axes and plots the clusters formed by them. 

    Args:
        X (array): Array containing pc1 in first column, pc2 in second column, etc. 
        n_clusters (int, optional): Number of clusters to form. Defaults to 2.

    Returns:
        array: array of labels, with 1 representing data points in the same cluster as the data from the first row. 
    """

    global labels, inv_labels
    
    # Make array of the first n principal components of the post data. 
    x_gauss = post_data['PC1']
    y_gauss = post_data['PC2']
    z_gauss = post_data['PC3']

    if num_dims=='2d':
        X_gauss = np.array([x_gauss, y_gauss]).T
        
    elif num_dims=='3d':
        X_gauss = np.array([x_gauss, y_gauss, z_gauss]).T  
    
    # Define constants
    X = X_gauss
    dims = X.shape[1]
    
    
    # Run Gaussian Mixture Model to obtain the two clusters. 
    gauss_mix = GaussianMixture(n_components=n_clusters)
    gm_fitted = gauss_mix.fit(X)
    labels = gm_fitted.predict(X)
    probabilities = gm_fitted.predict_proba(X)
    size = 10 * probabilities.max(axis=1) ** 2

    # Ensures improved patients are always marker as 1, and unimproved patients marked as zero
    # Relevant later
    if labels[0]==0:
        labels = abs(labels-1)
    
    # Unimproved patients now marked as 1
    inv_labels  = abs(labels-1)

    # Colours of clusters
    colour_list = ListedColormap(['red', 'deepskyblue'])

    # Plot
    if dims==2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap=colour_list, s=size, alpha=0.25)
        ax.scatter(X[0, 0], X[0, 1], s=size[0], marker='*', color='black')
    
    if dims==3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap=colour_list, s=size, alpha=0.15)
        ax.set_zlabel('PC3')

        
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Clusters formed from plotting PC1 against PC2')
    plt.show();
    
    return labels, inv_labels
    
    
def filter_patients(data, labels):
    """
    Filter data. Any data that corresponds to a zero in labels is converted to nan

    Args:
        data (array): Input data to be filtered
        labels (array): Array of 1s and 0s. 0 values get converted to a nan. 

    Returns:
        filtered_data (array): filtered data.
    """

    # Filter data 
    filtered_data = data * labels
    filtered_data[filtered_data==0] = np.nan 
    
    return filtered_data


def update_data_viewer(marker_num, view, viewing_options, pre_data, post_data, ax, fig, Improved_patients=True, Unimproved_patients=True, Patient_zero=False):
    """
    Updater function to allow the viewer to slide through the data for each marker.
    Shows the relationship between pre and post data, as well as allows viewer to see the distributions of the pre/post data separately. 
    Also allows viewing of improved/unimproved/patient zero or any combo of these options for each marker number

    Args:
        marker_num (int): Which marker the viewer wants to see. 
        histogram (str): Which data the viewer wants to see. 
    """
    
    # Creates array of the pre and post data for the required marker
    marker_i_pre = pre_data[f'marker_{marker_num}']
    marker_i_post = post_data[f'marker_{marker_num}']

    # Clears previous data. 
    ax.cla()
    
    improved_pre = filter_patients(marker_i_pre, labels)  
    improved_post = filter_patients(marker_i_post, labels)
    unimproved_pre = filter_patients(marker_i_pre, inv_labels)  
    unimproved_post = filter_patients(marker_i_post, inv_labels)  
    
    # Scatter plot of pre data and post data
    if view==viewing_options[0]:
        
        if Unimproved_patients==True:
            ax.scatter(unimproved_pre, unimproved_post, alpha=0.2, label='Unimproved patients', color='red') 
        if Improved_patients==True:
            ax.scatter(improved_pre, improved_post, alpha=0.2, label='Improved patients', color='deepskyblue') 
        if Patient_zero==True:
            ax.scatter(improved_pre[0], improved_post[0], color='black', marker='*', label='Patient Zero')   
             
        ax.set(xlim=(0.999*np.min(marker_i_pre), 1.001*np.max(marker_i_pre)), ylim=(0.999*np.min(marker_i_post), 1.003*np.max(marker_i_post)))
        ax.set_xlabel('Pre data')
        ax.set_ylabel('Post data')
        ax.legend(loc='upper left')
        
    # Histogram to show distribution of post data  
    elif view==viewing_options[1]:
        
        bins=100
        improved_post = improved_post[~np.isnan(improved_post)] 
        unimproved_post = unimproved_post[~np.isnan(unimproved_post)] 
        
        counts1  = np.histogram(improved_post, bins=bins)[0]
        counts2 = np.histogram(unimproved_post, bins=bins)[0]
    
        max_counts_arr = [np.max(counts1), np.max(counts2)]
        max_counts = np.max( max_counts_arr )
                
        if Unimproved_patients==True:
            ax.hist(unimproved_post, bins=bins, orientation='horizontal', label='Unimproved patients', color='red', alpha=0.2)
        if Improved_patients==True:
            ax.hist(improved_post, bins=bins, orientation='horizontal', label='Improved patients', color='deepskyblue', alpha=0.2)
        ax.set_xlabel('Frequency')   
        ax.legend()
        ax.set(xlim=(0, 1.1*max_counts), ylim=(0.999*np.min(marker_i_post), 1.003*np.max(marker_i_post)))
        
        
    ax.set_ylabel('Post data')   
    ax.set_title(f'Marker {marker_num}')
    ax.grid()
    ax.set_axisbelow(True)

    fig.canvas.draw()
    
    
def update_communality(num_components, loading_matrix, marker_names, width, ax, fig):
    """
    Updater function to calculate and plot the communality value up to a certain number of principal components for each marker

    Args:
        num_components (int): Number of principal components. 
    """
    
    ax.cla()
    
    # Sum 
    sum = 100 * np.sum(loading_matrix.iloc[:, :num_components]**2, axis=1)

    ax.bar(marker_names, sum, width,)
    
    # Figure setup
    ax.set_xlabel(f'{num_components} components')
    ax.set(ylim=(0, 100) )
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_xlabel('Markers')
    ax.set_ylabel('Communality / %')

    fig.canvas.draw()
    