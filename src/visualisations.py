import numpy as np
import matplotlib.pyplot as plt
from src.data_processing import prepared_df_nruns_barplot, peak_position

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          #'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

def val_eff_plots(df1, df2):
    """
    Generate subplots of validation and efficiency plots for different confidence levels.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data.

    Returns:
        None
    """
    # Extract the required information for different confidence levels
    #RFCP_prediction_70 = df[df['CL'] == 0.3].groupby('runs', as_index=False).mean()
    #RFCP_prediction_75 = df[df['CL'] == 0.25].groupby('runs', as_index=False).mean()
    #RFCP_prediction_80 = df[df['CL'] == 0.2].groupby('runs', as_index=False).mean()
    #RFCP_prediction_85 = df[df['CL'] == 0.15].groupby('runs', as_index=False).mean()
    RFCP_prediction_90_1 = df1[df1['CL'] == 0.1].groupby('runs', as_index=False).mean()
    RFCP_prediction_90_2 = df2[df2['CL'] == 0.1].groupby('runs', as_index=False).mean()

    #RFCP_prediction_95 = df[df['CL'] == 0.05].groupby('runs', as_index=False).mean()

    print(f"90CL Model17: {len(RFCP_prediction_90_1.loc[RFCP_prediction_90_1['validity']>=0.9])}, average = {np.mean(RFCP_prediction_90_1['validity'])} and avg efficiency = {np.mean(RFCP_prediction_90_1['efficiency'])}")
    print(f"90CL Model33: {len(RFCP_prediction_90_2.loc[RFCP_prediction_90_2['validity']>=0.9])}, average = {np.mean(RFCP_prediction_90_2['validity'])} and avg efficiency = {np.mean(RFCP_prediction_90_2['efficiency'])}")
    #print(f"80CL: {len(RFCP_prediction_80.loc[RFCP_prediction_80['validity']>=0.8])}, average = {np.mean(RFCP_prediction_80['validity'])} and avg efficiency = {np.mean(RFCP_prediction_80['efficiency'])}")
    #print(f"70CL: {len(RFCP_prediction_70.loc[RFCP_prediction_70['validity']>=0.7])}, average = {np.mean(RFCP_prediction_70['validity'])} and avg efficiency = {np.mean(RFCP_prediction_70['efficiency'])}")

    # Create a 3x2 grid of subplots with shared x and y axes
    #f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex='all', sharey='all', figsize=(20, 20))
    f, (ax1, ax2) = plt.subplots(1, 2, sharex='all', sharey='all', figsize=(13, 6))
    x = np.array([z for z in range(1, 101)])

    # Plot the data for each confidence level subplot
    ax1.plot(x, np.array(RFCP_prediction_90_1.efficiency.tolist()),'o-', color='r', label='Efficiency')
    ax1.plot(x, np.array(RFCP_prediction_90_1.validity.tolist()),'.-', color='b', label='Validity')
    ax1.axhline(y=0.90, color = 'green')
    ax1.grid(axis = 'y')
    ax1.set_title('Model17', fontsize=20)
    ax1.locator_params(axis='y', nbins = 6)
    ax1.tick_params(axis='both', labelsize=16)
    ax1.set_xlabel('Number of Split', fontsize=18)
    ax1.set_ylabel('Validity/Efficiency', fontsize=18)

    ax2.plot(x, np.array(RFCP_prediction_90_2.efficiency.tolist()),'o-', color='r', label='Efficiency')
    ax2.plot(x, np.array(RFCP_prediction_90_2.validity.tolist()),'.-', color='b', label='Validity')
    ax2.axhline(y=0.90, color = 'green')
    ax2.grid(axis = 'y')
    ax2.set_title('Model33', fontsize=20)
    ax2.tick_params(axis='both', labelsize=18)
    ax2.set_xlabel('Number of Split', fontsize=18)

    ax2.legend(fontsize = 18)

    plt.show()

def plot_hist(ax, TC, end_plot = False):
    """
    Plot a histogram of Tanimoto coefficients with colored bars.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        TC (list or np.ndarray): List of Tanimoto coefficients.

    Returns:
        matplotlib.axes.Axes: The axes with the histogram plot.
    """
    # Plot a histogram with 100 bins in the range [0.0, 1.0]
    _, _, patches = ax.hist(TC, bins=100, range=[0.0, 1.0], alpha=0.7)

    # Add a vertical dashed line at the mean of Tanimoto coefficients in green
    ax.axvline(np.asarray(TC).mean(), color='green', linestyle='dashed', linewidth=1)

    ax.axvline(0.4, color='black', linestyle='dashed', linewidth=1)
    ax.axvline(0.7, color='black', linestyle='dashed', linewidth=1)


    # Color the first 40 bins' patches as red, next 30 as blue, and last 30 as green
    for i in range(0, 40):
        patches[i].set_facecolor('r')
    for i in range(40, 70):
        patches[i].set_facecolor('blue')
    for i in range(70, 100):
        patches[i].set_facecolor('green')

    # Customize the plot appearance and labels
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    if end_plot:
        ax.set_xlabel('Highest Tc', fontsize=18)
    ax.grid(True)

    return ax

def stacked_barplot_MCCP_predictions(df1, df2):
    """
    Create a stacked barplot to compare predictions from two DataFrames.

    Args:
        df1 (pd.DataFrame): The first DataFrame with prediction data.
        df2 (pd.DataFrame): The second DataFrame with prediction data.

    Returns:
        None
    """
    # Create subplots with shared x-axis
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 9), sharex=True)

    # Plot data on the subplots
    ax1 = prepared_df_nruns_barplot(df1).plot(kind='bar', stacked=True, ax=axes[0], legend=False)
    ax1.tick_params(axis='both', labelsize=20)
    ax2 = prepared_df_nruns_barplot(df2).plot(kind='bar', stacked=True, ax=axes[1], legend=False)
    ax2.tick_params(axis='both', labelsize=20)

    # Add titles and adjust font sizes
    ax1.set_title('A', fontsize=30)
    ax2.set_title('B', fontsize=30)

    # Set common x-axis label

    # Set common x-axis ticks and labels on the bottom subplot
    # plt.xticks(ax2.get_xticks(), ax2.get_xticklabels(), fontsize=12)
    x_ticks = ax2.get_xticks()[::5]
    x_ticklabels = ax2.get_xticklabels()[::5]
    plt.xticks(x_ticks, x_ticklabels, fontsize=20)
    plt.yticks(fontsize=20)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_hist_1_2(ax, TC1, TC2, title, TS_thr):
    """
    Plot histograms of two sets of Tanimoto coefficients (TC1 and TC2) and threshold values.

    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): The subplot to plot the histograms.
        TC1 (list): List of Tanimoto coefficients for the first set (e.g., Model17).
        TC2 (list): List of Tanimoto coefficients for the second set (e.g., Model33).
        title (str): The title for the plot.
        TS_thr (list): A list containing two threshold values.

    Returns:
        matplotlib.axes._subplots.AxesSubplot: The modified subplot with histograms and threshold lines.
    """
    # Plot histograms for TC1 and TC2 with different colors and transparency
    ax.hist(TC1, bins=100, range=[0.0, 1.0], alpha=0.5, color='blue', label='Model17')
    ax.hist(TC2, bins=100, range=[0.0, 1.0], alpha=0.5, color='orange', label='Model33')

    # Add dashed lines for the mean values of TC1 and TC2
    ax.axvline(np.asarray(TC1).mean(), color='darkblue', linestyle='dashed', linewidth=1)
    ax.axvline(np.asarray(TC2).mean(), color='darkorange', linestyle='dashed', linewidth=1)

    # Add threshold lines with a specified color and linewidth
    ax.axvline(TS_thr[0], color='lightcoral', linewidth=2)
    ax.axvline(TS_thr[1], color='lightcoral', linewidth=2)

    # Set the title, grid, and legend for the plot
    ax.set_title(title, fontsize=26)
    ax.grid(True)
    ax.legend()

    return ax

def plot_coverage_analysis(ax, TC1, TC2, TC3, title, legend = False, y_label = False):
    # Plot histograms for TC1 and TC2 with different colors and transparency
    # Compute histograms
    hist_TC1, bins_TC1 = np.histogram(TC1, bins=100, range=[0.0, 0.7], density=True)
    hist_TC2, bins_TC2 = np.histogram(TC2, bins=100, range=[0.0, 0.7], density=True)
    hist_TC3, bins_TC3 = np.histogram(TC3, bins=100, range=[0.0, 0.7], density=True)
    #peak_position(TC1)
    #peak_position(TC2)
    #peak_position(TC3)

    find_peak(TC1)
    find_peak(TC2)
    find_peak(TC3)

    # Generate x values for the histogram curves
    x_TC1 = (bins_TC1[:-1] + bins_TC1[1:]) / 2
    x_TC2 = (bins_TC2[:-1] + bins_TC2[1:]) / 2
    x_TC3 = (bins_TC3[:-1] + bins_TC3[1:]) / 2

    # Plot the histogram curves
    ax.plot(x_TC1, hist_TC1, color='blue', label='Low similarity')
    ax.plot(x_TC2, hist_TC2, color='green', label='Medium similarity')
    ax.plot(x_TC3, hist_TC3, color='red', label='High similarity')

    # Add dashed lines for the mean values of TC1, TC2, TC3
    #ax.axvline(np.mean(TC1), color='blue', linestyle='dashed', linewidth=1)
    #ax.axvline(np.mean(TC2), color='green', linestyle='dashed', linewidth=1)
    #ax.axvline(np.mean(TC3), color='red', linestyle='dashed', linewidth=1)

    # Set the title, grid, and legend for the plot
    ax.set_title(title, fontsize=20)
    ax.set_xlabel('Highest Tc', fontsize=18)
    if y_label:
        ax.set_ylabel('Frequency', fontsize=18)
    ax.grid(True)
    if legend:
        ax.legend()

    return ax
    
    """ax.hist(TC1, bins=100, range=[0.0, 0.7], density = True, alpha=0, color='lightblue', label='Low similarity')
    ax.hist(TC2, bins=100, range=[0.0, 0.7], density = True, alpha=0, color='lightgreen', label='Medium similarity')
    ax.hist(TC3, bins=100, range=[0.0, 0.7], density = True, alpha=0, color='coral', label='High similarity')
    peak_position(TC1)
    peak_position(TC2)
    peak_position(TC3)

    # Add dashed lines for the mean values of TC1 and TC2
    ax.axvline(np.asarray(TC1).mean(), color='blue', linestyle='dashed', linewidth=1)
    ax.axvline(np.asarray(TC2).mean(), color='green', linestyle='dashed', linewidth=1)
    ax.axvline(np.asarray(TC3).mean(), color='red', linestyle='dashed', linewidth=1)

    # Add threshold lines with a specified color and linewidth
    #ax.axvline(TS_thr[0], color='lightcoral', linewidth=2)
    #ax.axvline(TS_thr[1], color='lightcoral', linewidth=2)

    # Set the title, grid, and legend for the plot
    ax.set_title(title, fontsize=26)
    ax.set_xlabel('Highest Tanimoto coefficient', fontsize=20)
    ax.grid(True)
    if legend:
        ax.legend()

    return ax"""

from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score

def gaussian(x, amplitude, mean, stddev):
    """
    Gaussian function.
    """
    return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)

def find_peak(data, fit_range = (10,30)):
    # Create a histogram
    hist, bins = np.histogram(data, bins=100, range=fit_range)

    # Calculate bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Fit a Gaussian curve to the histogram
    popt, _ = curve_fit(gaussian, bin_centers, hist, p0=[1.0, np.mean(data), np.std(data)])

    # Calculate the position of the peak (mean of the Gaussian)
    peak_position = popt[1]

    # Extract the parameters of the Gaussian curve
    #amplitude, mean, stddev = popt

    # Calculate goodness-of-fit metrics
    #predicted_values = gaussian(bin_centers, amplitude, mean, stddev)
    #r_squared = r2_score(hist, predicted_values)
    #mse = mean_squared_error(hist, predicted_values)

    # Print the position of the peak
    #print('Position of the peak:', peak_position, r_squared, mse)
    print('Position of the peak:', peak_position)

def KL_histograms(data1, data2):

    
    # Plot the histograms
    plt.hist(data1, bins=100, range=[0.0, 1.0], alpha=0.5, color='blue', label='GRML library')
    plt.hist(data2, bins=100, range=[0.0, 1.0], alpha=0.5, color='orange', label='RML library')

    # Calculate and plot vertical lines for the average values
    plt.axvline(np.asarray(data1).mean(), color='blue', linestyle='dashed', linewidth=1)
    plt.axvline(np.asarray(data2).mean(), color='orange', linestyle='dashed', linewidth=1)

    plt.xlabel('Highest Tc', fontsize = 18)
    plt.ylabel('Frequency' , fontsize = 18)
    plt.title('Novel set vs. GRML and RML', fontsize = 20)
    plt.grid(True)
    plt.legend()
    plt.show()