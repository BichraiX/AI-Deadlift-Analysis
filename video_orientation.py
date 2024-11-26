def deriv_Barre(yBarre, i, smoothing = 5):
    """
    Calculate the derivative (rate of change) of the bar's vertical position over a specified number of frames.

    Parameters:
    - yBarre (list or array): Vertical positions of the bar over time.
    - i (int): The current frame index at which to compute the derivative.
    - smoothing (int): Number of frames over which to compute the derivative, used to reduce noise.

    Returns:
    - float: The rate of change (derivative) of yBarre at frame i.
    """
    return (yBarre[i + smoothing] - yBarre[i]) / smoothing

def is_vertical(yBarre, yGenou):
    """
    Determine if the bar starts below the knees at the beginning of the movement.

    Parameters:
    - yBarre (list or array): Vertical positions of the bar over time.
    - yGenou (list or array): Vertical positions of the knees over time.

    Returns:
    - bool: True if the bar's initial position is below the knees, otherwise False.
    """
    if yBarre[0] < yGenou[0]:
        return True
    return False

def bar_direction(yBarre, i, smoothing, y1, y2, alpha=0.01):
    """
    Determine the direction of the bar's movement (up, down, or still) at a specific point in time.

    Parameters:
    - yBarre (list or array): Vertical positions of the bar over time.
    - i (int): The current frame index at which to check the bar's movement.
    - smoothing (int): Number of frames over which to compute the derivative.
    - y1 (list or array): Vertical positions of the first body part (e.g., shoulders).
    - y2 (list or array): Vertical positions of the second body part (e.g., waist).
    - alpha (float): Threshold ratio for determining significant movement. The derivative must exceed
                     alpha times the vertical distance between y1 and y2 to count as movement.

    Returns:
    - str: "up" if the bar is moving upward.
           "down" if the bar is moving downward.
           "still" if the bar is stationary.
    """
    treshold = abs(y2[i] - y1[i]) * alpha  # Threshold based on the vertical distance between y1 and y2
    if deriv_Barre(yBarre, i, smoothing) > treshold:
        return "up"
    elif deriv_Barre(yBarre, i, smoothing) < -treshold:
        return "down"
    else:
        return "still"