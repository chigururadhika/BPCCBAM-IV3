import matplotlib.pyplot as plt
import numpy as np

def plot_model_results(results,metric_name=None, model_names=None, title=None, xlim=None, ylim=None):
    """
    Plot the results of different models as bar graphs.

    Args:
    - results (list): A list of tuples, each containing (model_name, metric_value).
    - metric_names (list, optional): List of metric names corresponding to the results.
    - title (str, optional): Title for the bar graph.
    - xlim (tuple, optional): Tuple (x_min, x_max) for custom x-axis limits.
    - ylim (tuple, optional): Tuple (y_min, y_max) for custom y-axis limits.

    Returns:
    - None: Displays the bar graph.
    """
    if not results:
        print("No results to plot.")
        return

    model_names, metric_values = zip(*results)
    x = np.arange(len(model_names))

    plt.figure(figsize=(8, 6))
    for i,j in zip(model_names, metric_values):    
        plt.bar(i, j, align='center', alpha=0.7)
        plt.xticks(x, model_names, rotation='vertical')
        plt.ylabel(metric_name)
        plt.xlabel('model')

    if model_names:
        plt.legend(model_names, loc='upper right')

    if title:
        plt.title(title)

    if xlim:
        plt.xlim(xlim)

    if ylim:
        plt.ylim(ylim)

    plt.tight_layout()
    

# Example usage:
