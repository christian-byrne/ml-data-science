import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Define a Sample Cost Function ---
def cost_function(x, y):
    """
    Example cost function: A simple quadratic bowl.
    """
    return x**2 + y**2

def gradient(x, y):
    """
    Gradient of the cost function.
    """
    grad_x = 2 * x
    grad_y = 2 * y
    return np.array([grad_x, grad_y])

# --- Generate Data for Mini-Batches ---
def generate_data(num_samples, x_range, y_range):
    """
    Generate random data points within a specified range for demonstration.
    
    Parameters:
        num_samples (int): Number of data points.
        x_range, y_range (tuple): Range of x and y values.
    
    Returns:
        np.ndarray: Array of data points (x, y).
    """
    x = np.random.uniform(x_range[0], x_range[1], num_samples)
    y = np.random.uniform(y_range[0], y_range[1], num_samples)
    return np.column_stack((x, y))

# --- Mini-Batch Gradient Descent Implementation ---
def mini_batch_gradient_descent(data, learning_rate, num_steps, batch_size):
    """
    Perform mini-batch gradient descent to minimize the cost function.
    
    Parameters:
        data (np.ndarray): Array of data points (x, y).
        learning_rate (float): Step size for gradient descent.
        num_steps (int): Number of steps to run gradient descent.
        batch_size (int): Size of each mini-batch.
    
    Returns:
        list: Points visited during gradient descent.
    """
    np.random.seed(42)  # For reproducibility
    current = np.array([np.mean(data[:, 0]), np.mean(data[:, 1])])  # Start at the data mean
    points = [current]

    for _ in range(num_steps):
        # Select a random mini-batch
        indices = np.random.choice(len(data), size=batch_size, replace=False)
        batch = data[indices]

        # Compute the gradient for the batch
        grad = np.mean([gradient(point[0], point[1]) for point in batch], axis=0)
        
        # Update the current point
        current = current - learning_rate * grad
        points.append(current)
    
    return points

# --- Visualization Utilities ---
def plot_cost_function(ax, x_range, y_range):
    """
    Plot the cost function as a contour map.
    
    Parameters:
        ax: Matplotlib Axes object.
        x_range, y_range: Ranges for x and y values.
    """
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = cost_function(X, Y)
    ax.contour(X, Y, Z, levels=30, cmap='viridis')
    ax.set_title("Mini-Batch Gradient Descent")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return X, Y, Z

def visualize_mini_batch_gradient_descent(data, points, x_range, y_range, batch_size):
    """
    Visualize the mini-batch gradient descent process.
    
    Parameters:
        data (np.ndarray): Array of data points (x, y).
        points (list): Points visited during gradient descent.
        x_range, y_range: Ranges for x and y values.
        batch_size (int): Size of each mini-batch.
    """
    points = np.array(points)

    # Set up plot
    fig, ax = plt.subplots(figsize=(8, 6))
    X, Y, Z = plot_cost_function(ax, x_range, y_range)

    # Plot the path of gradient descent
    path, = ax.plot([], [], 'ro-', markersize=5, label='Path')
    current_point, = ax.plot([], [], 'bo', label='Current Point')

    # Animate the gradient descent
    def update(frame):
        if frame < len(points):
            path.set_data(points[:frame, 0], points[:frame, 1])
            current_point.set_data(points[frame, 0], points[frame, 1])
        return path, current_point

    ani = FuncAnimation(fig, update, frames=len(points), interval=200, blit=True)

    # Add legend and show the animation
    ax.legend()
    plt.title(f"Mini-Batch Size: {batch_size}")
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Parameters for the cost function
    x_range = (-3, 3)
    y_range = (-3, 3)

    # Generate data points
    num_samples = 100
    data = generate_data(num_samples, x_range, y_range)

    # Parameters for gradient descent
    learning_rate = 0.1
    num_steps = 50
    batch_size = 10  # Adjust this to see how batch size affects optimization

    # Perform mini-batch gradient descent
    points = mini_batch_gradient_descent(data, learning_rate, num_steps, batch_size)

    # Visualize the process
    visualize_mini_batch_gradient_descent(data, points, x_range, y_range, batch_size)
