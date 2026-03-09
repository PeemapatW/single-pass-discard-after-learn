import numpy as np


def get_axis_edge_points(center, axes, widths):
    """Generates edge points for each axis."""
    edge_points = []
    for i in range(len(axes)):
        edge_points.append([center - widths[i] * axes[i],center + widths[i] * axes[i]])
    return np.array(edge_points)

def get_axis_sample_points(center, axes, widths, num_samples=11):
    """
    Generates a specified number of sample points along each principal axis.

    This function is used to create a set of points for robustly checking
    if an axis of one hyper-ellipsoid intersects with another.

    Args:
        center (np.ndarray): The center vector of the hyper-ellipsoid.
        axes (np.ndarray): A 2D array where each row is an eigenvector (principal axis).
        widths (np.ndarray): A 1D array of the semi-axis lengths (r * sqrt(eigenvalue)).
        num_samples (int, optional): The number of points to sample along each axis.
                                    Defaults to 11. An odd number is recommended
                                    to include the center point.

    Returns:
        list: A list of numpy arrays. Each array contains the `num_samples`
                points sampled along the corresponding principal axis.
    """
    all_axis_points = []
    # Ensure axes and widths are iterable
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    if not isinstance(widths, (list, np.ndarray)):
        widths = [widths]

    for i in range(len(axes)):
        # Define the start and end points of the axis segment
        start_point = center - widths[i] * axes[i]
        end_point = center + widths[i] * axes[i]

        # Use np.linspace to generate num_samples evenly spaced points
        # between the start and end points.
        # The 'axis=1' argument stacks the start and end points for broadcasting.
        axis_points = np.linspace(start_point, end_point, num_samples)

        all_axis_points.append(axis_points)

    return all_axis_points
