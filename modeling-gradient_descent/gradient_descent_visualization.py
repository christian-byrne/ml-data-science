import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Define a Sample Cost Function ---
def cost_function(x, y):
    """
    Example cost function: A simple quadratic bowl with a saddle-like structure.
    """
    return x**2 + y**2

def gradient(x, y):
    """
    Gradient of the cost function.
    """
    grad_x = 2 * x
    grad_y = 2 * y
    return np.array([grad_x, grad_y])

# --- Gradient Descent Implementation ---
def gradient_descent(start, learning_rate, num_steps):
    """
    Perform gradient descent to minimize the cost function.
    
    Parameters:
        start (tuple): Starting point (x, y).
        learning_rate (float): Step size for gradient descent.
        num_steps (int): Number of steps to run gradient descent.
    
    Returns:
        list: Points visited during gradient descent.
    """
    points = [np.array(start)]
    for _ in range(num_steps):
        current = points[-1]
        grad = gradient(current[0], current[1])
        next_point = current - learning_rate * grad
        points.append(next_point)
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
    ax.set_title("Gradient Descent Path")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return X, Y, Z

def visualize_gradient_descent(start, learning_rate, num_steps, x_range, y_range):
    """
    Visualize the gradient descent process.
    
    Parameters:
        start (tuple): Starting point (x, y).
        learning_rate (float): Step size for gradient descent.
        num_steps (int): Number of steps to run gradient descent.
        x_range, y_range: Ranges for x and y values.
    """
    # Perform gradient descent
    points = gradient_descent(start, learning_rate, num_steps)
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

    ani = FuncAnimation(fig, update, frames=num_steps + 1, interval=200, blit=True)

    # Add legend and show the animation
    ax.legend()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Parameters for gradient descent
    start = (3, 3)         # Starting point
    learning_rate = 0.1    # Step size
    num_steps = 50         # Number of steps

    # Visualization ranges
    x_range = (-4, 4)
    y_range = (-4, 4)

    # Visualize gradient descent
    visualize_gradient_descent(start, learning_rate, num_steps, x_range, y_range)
