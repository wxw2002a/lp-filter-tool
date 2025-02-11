import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# A very small constant used to avoid division by zero in computations.
EPSILON = 1e-18

def interpolate_quadratic_vertex(x_vals, y_vals):
    """
    Fit a quadratic polynomial to the provided candidate points and return the x-coordinate of its vertex.

    This function takes in an array of candidate x-values (e.g., possible output values)
    and their corresponding error values (y_vals), fits a quadratic (second order polynomial)
    to these points, and then calculates the vertex (minimum) of the quadratic using the formula:
        vertex = -b / (2a)
    where a and b are the first two coefficients of the fitted polynomial.

    Parameters:
        x_vals (array-like): Candidate x-values.
        y_vals (array-like): Corresponding error (or cost) values for each x-value.

    Returns:
        float: The x-coordinate of the vertex (i.e., the estimated optimal candidate).

    Raises:
        ValueError: If the quadratic coefficient (a) is nearly zero, which would make the vertex undefined.
    """
    p = np.polyfit(x_vals, y_vals, 2)  # Fit a quadratic: p[0]*x^2 + p[1]*x + p[2]
    if abs(p[0]) < EPSILON:
        raise ValueError("Quadratic coefficient is near zero, cannot find a unique vertex.")
    return -p[1] / (2 * p[0])  # Calculate and return the vertex of the quadratic

def generate_signal(signal_type, n=100, t1=3, t2=0.005):
    """
    Generate a signal of a specified type: 'step', 'ramp', or 'sin'.

    Depending on the value of signal_type, this function creates a time vector and generates:
      - 'step': A step function that switches from 0 to 1 at time t2.
      - 'ramp': A ramp (linearly increasing function) until time t2, then remains constant.
      - 'sin': A sinusoidal signal (with frequency 1 Hz) over a longer duration for smoothness.

    Parameters:
        signal_type (str): The type of signal to generate ('step', 'ramp', or 'sin').
        n (int): The number of samples per unit time (sampling rate). Default is 100.
        t1 (float): The duration for which the time vector is created (in seconds). Default is 3.
        t2 (float): The threshold time for switching/transition in step and ramp signals. Default is 0.005.

    Returns:
        np.array: The generated signal array.

    Raises:
        ValueError: If an unsupported signal_type is provided.
    """
    # Create a time vector from 0 to t1 with a time step of 1/n seconds.
    t = np.arange(0, t1 + 1/n, 1/n)
    if signal_type == 'step':
        # For a step signal, return an array that is 0 when t < t2 and 1 when t >= t2.
        return (t >= t2).astype(float)
    elif signal_type == 'ramp':
        # Create a ramp signal: increase linearly until time t2, then hold constant.
        nx = round(t2 * n)  # Number of samples corresponding to time t2.
        x_in = np.zeros_like(t)
        x_in[:nx] = t[:nx]         # Linear ramp for the first nx samples.
        x_in[nx:] = t[nx - 1]        # Constant value after the ramp.
        return x_in
    elif signal_type == 'sin':
        # Generate a sinusoidal signal with a frequency of 1 Hz.
        freq = 1
        # For the sine wave, create a longer time vector for a smoother waveform.
        t_sin = np.arange(0, 100.01, 0.01)
        return np.sin(2 * np.pi * freq * t_sin)
    else:
        raise ValueError(f"Unsupported signal_type: {signal_type}")

def read_csv_signal(filepath, column=1, max_samples=None):
    """
    Read a CSV file and extract a single column of data as a NumPy array.

    The CSV file is read assuming it has no header (or the header is not required).
    The function extracts the specified column (0-indexed) and, if desired,
    limits the number of samples returned.

    Parameters:
        filepath (str): The path to the CSV file.
        column (int): The column index to extract (default is 1).
        max_samples (int or None): If provided, limits the returned array to the first max_samples elements.

    Returns:
        np.array: An array of the extracted column data.
    """
    df = pd.read_csv(filepath, header=None)
    data = df.iloc[:, column].to_numpy()  # Extract the specified column.
    if max_samples is not None:
        data = data[:max_samples]  # Limit the number of samples if needed.
    return data

def R_calculation(x, y, p, T, h):
    """
    Compute the R(y) value needed for the Lp filter update.

    For a given window of input samples x and a candidate filter output y,
    this function computes a weighted combination of the differences between
    the input and y. Two components are computed:
      - s1: A contribution from the last sample in x.
      - s2: A summation over the remaining samples.
    The combination s1+s2 is then used to compute the R value based on the Lp norm.

    Parameters:
        x (np.array): Array of input samples (a window of past inputs).
        y (float): The candidate output value.
        p (float): The Lp norm parameter.
        T (float): The period or time constant used in the exponential weighting.
        h (float): The sampling time interval.

    Returns:
        float: The computed R(y) value. Returns 0.0 if the intermediate sum is nearly zero.
    """
    n = len(x)
    # Precompute exponential weights for each sample in the window.
    exp_vals = np.exp(np.arange(1, n + 1) * h / T)

    # Compute s1 using the last sample in x.
    s1 = (h * T / 2.0) * (abs(x[-1] - y)**(p - 1)) * np.sign(x[-1] - y)
    s1 *= exp_vals[-1]  # Multiply by the exponential weight for the last sample.

    # Compute s2 by summing over all samples except the last one.
    diff = x[:-1] - y
    s2_terms = (np.abs(diff)**(p - 1)) * np.sign(diff) * exp_vals[:-1]
    s2 = np.sum(s2_terms) * (h / T)

    s21 = s1 + s2  # Combined sum from both contributions.
    
    # If the sum is nearly zero, return 0 to avoid issues with fractional exponents.
    if abs(s21) < EPSILON:
        return 0.0

    # Return the scaled R value based on the Lp norm.
    return (abs(s21)**(1 / (p - 1))) * np.sign(s21)

def lp_filter(
    Lp_norm_number,
    signal='step',
    N=300,
    n=500,
    y_init=0,
    y_delta=0.01,
    y_beta=0.001,
    h=0.01,
    T=1
):
    """
    Compute the output of an Lp low-pass filter given an input signal.

    This function implements an iterative filter update algorithm. At each time step,
    it uses a sliding window of input data (of length n) and computes the next filter output
    by comparing three candidate values: the previous output, and small deviations below and above
    that value. For each candidate, it computes a corresponding R value (using R_calculation) and
    selects an optimal candidate using quadratic interpolation. A small refinement step further fine-tunes
    the selected output.

    Parameters:
        Lp_norm_number (float): The Lp norm parameter (p value) used in the filter calculation.
        signal (str): The type of input signal to filter ('step', 'ramp', or 'sin'). Default is 'step'.
        N (int): The total number of output samples to compute. Default is 300.
        n (int): The number of past input samples used in each filter update (window length). Default is 500.
        y_init (float): The initial filter output value. Default is 0.
        y_delta (float): The small deviation used to generate candidate outputs around the current value. Default is 0.01.
        y_beta (float): The increment used in the refinement step to further adjust the candidate output. Default is 0.001.
        h (float): The sampling time interval. Default is 0.01.
        T (float): The period or time constant for the exponential weighting in R_calculation. Default is 1.

    Returns:
        tuple: A tuple containing:
            - y (np.array): The filtered output signal.
            - x_in (np.array): The input signal used (truncated to N samples if needed).
    """
    # Generate the input signal based on the specified type.
    if signal == 'step':
        x_in = generate_signal('step')
    elif signal == 'ramp':
        x_in = generate_signal('ramp')
    else:
        # If not 'step' or 'ramp', default to generating a sinusoidal signal.
        x_in = generate_signal('sin')

    # Adjust N if the input signal is shorter than the desired number of samples.
    if len(x_in) < N:
        N = len(x_in)

    # Initialize the output signal array with zeros and set the first value.
    y = np.zeros(N, dtype=float)
    y[0] = y_init

    p = Lp_norm_number  # The p parameter for the Lp norm

    # Iterate over each time step to update the filter output.
    for step in range(1, N):
        # Create a sliding window of input samples (of length n) for the current step.
        if step < n:
            # For early steps, pad the beginning of the window with zeros.
            x_window = np.zeros(n)
            x_window[n - step - 1:] = x_in[:step + 1]
        else:
            # Use the last n samples of the input signal.
            x_window = x_in[step - n + 1 : step + 1]

        # Use the previous output value as the starting point.
        y2 = y[step - 1]
        # Define two nearby candidate output values: one slightly lower and one slightly higher.
        y1, y3 = y2 - y_delta, y2 + y_delta
        y_candidates = np.array([y1, y2, y3])

        # Compute the R value for each candidate.
        R_vals = np.array([R_calculation(x_window, cand, p, T, h) for cand in y_candidates])
        # Compute the squared error between each candidate and its R value.
        j = (y_candidates - R_vals)**2
        # Use quadratic interpolation to estimate the optimal candidate output.
        y_out = interpolate_quadratic_vertex(y_candidates, j)

        # --- Refinement Step ---
        # Define a function to compute the absolute difference between a candidate value and its R value.
        def diff_fn(val):
            return abs(val - R_calculation(x_window, val, p, T, h))

        diff_old = diff_fn(y_out)
        # Attempt a small increment to see if the error decreases.
        y_out_pos = y_out + y_beta
        diff_new = diff_fn(y_out_pos)

        if diff_new < diff_old:
            # Increase y_out incrementally until no further improvement is found.
            while diff_new < diff_old:
                y_out = y_out_pos
                diff_old = diff_new
                y_out_pos += y_beta
                diff_new = diff_fn(y_out_pos)
        else:
            # If increasing did not reduce the error, try decreasing y_out.
            y_out_neg = y_out - y_beta
            diff_new = diff_fn(y_out_neg)
            while diff_new < diff_old:
                y_out = y_out_neg
                diff_old = diff_new
                y_out_neg -= y_beta
                diff_new = diff_fn(y_out_neg)

        # Save the refined output value for the current time step.
        y[step] = y_out

    return y, x_in[:N]

def plot_lp_filter():
    """
    Plot the responses of the Lp filter for various p values using a step input signal.

    This function serves as an example routine that:
      1. Generates a step input signal.
      2. Computes the Lp filter outputs for a range of Lp norm parameters.
      3. Plots the input signal and the corresponding filtered outputs in different colors.
    
    Usage:
        When this script is run directly (i.e., not imported as a module),
        this function will be executed and display a plot window with the results.
    """
    # List of Lp norm parameters (p values) for which to compute the filter output.
    lp_values = [2, 1.8, 1.6, 1.4, 1.2, 1.1, 1.02]
    # Define a set of colors for plotting: first color for the input signal, and others for each lp value.
    colors = ["#E0E0E0", "#CC0000", "#FF8000", "#FFFF00", "#00FF00", "#00FFFF", "#0080FF", "#FF00FF"]

    plt.figure()
    # Generate the input signal using a default Lp value (2) to extract and plot the input.
    _, x_in = lp_filter(2, signal='step')
    plt.plot(x_in, color=colors[0], label="Input")

    # Loop over the different Lp values, compute the filtered output, and plot each.
    for i, lp in enumerate(lp_values):
        y_out, _ = lp_filter(lp, signal='step')
        plt.plot(y_out, color=colors[i+1], label=f"lp={lp:.2f}")

    plt.legend(loc="best")
    plt.title("Lp Filter Responses")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.show()

# If this script is executed as the main program, run the example plot.
if __name__ == "__main__":
    plot_lp_filter()
