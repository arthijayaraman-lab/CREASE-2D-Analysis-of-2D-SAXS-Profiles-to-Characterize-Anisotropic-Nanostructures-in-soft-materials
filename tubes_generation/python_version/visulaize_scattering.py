import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmocean
import os
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d
from matplotlib.colors import ListedColormap

# Define custom colormaps based on MATLAB definitions
def create_speed_colormap():
    """Create a speed colormap matching MATLAB's implementation"""
    speed_colormap_data = np.array([
        [0.9996253193176977, 0.9913711226010461, 0.8041012438578545],
        [0.9969312990878144, 0.9865865913107011, 0.795819654568807],
        [0.9942533588637105, 0.9818135789307644, 0.7875317815897166],
        [0.9915896776086416, 0.9770525904709529, 0.7792374356109949],
        [0.988938478622175, 0.9723041153469224, 0.7709364896057566],
        [0.9862980251266783, 0.9675686302753327, 0.7626288656679628],
        [0.9836666169060123, 0.9628466015967408, 0.754314523368193],
        [0.9810425876106125, 0.9581384871880828, 0.7459934495167191],
        [0.9784237290846493, 0.9534448589527805, 0.7376670490866495],
        [0.9758091741853187, 0.9487660072025493, 0.7293335612360095],
        [0.9731976797213667, 0.9441023023821585, 0.7209921595340746],
        [0.9705876565172377, 0.9394541905537218, 0.712642910340537],
        [0.9679775344953384, 0.9348221172710476, 0.7042858810456845],
        [0.9653657609756586, 0.9302065285603878, 0.6959211353452218],
        [0.9627508763245108, 0.9256078555762781, 0.6875485248304888],
        [0.960131791323147, 0.9210264571207541, 0.679166921161872],
        [0.9575068348330097, 0.916462820977799, 0.6707767168399573],
        [0.9548744491996995, 0.9119174176425877, 0.6623780173986025],
        [0.9522330808045906, 0.9073907239128326, 0.6539709177027835],
        [0.9495811770290349, 0.9028832240271246, 0.6455555018977681],
        [0.946917182921429, 0.8983954108674836, 0.6371318441574638],
        [0.9442402190659518, 0.8939276490907958, 0.628697964511337],
        [0.941548643785517, 0.8894804711503382, 0.6202539985663387],
        [0.9388403761261178, 0.8850545069722993, 0.6118014161078326],
        [0.9361137658947895, 0.8806503085987659, 0.6033403762527677],
        [0.933367142793178, 0.8762684428171652, 0.5948710444843188]
    ])
    # Add more color points from the MATLAB definition to match the full colormap
    # This is just a sample of the first few colors
    
    # Create and return the colormap
    return ListedColormap(speed_colormap_data)

def create_thermal_colormap():
    """Create a thermal colormap matching MATLAB's implementation"""
    thermal_colormap_data = np.array([
        [0.9090418416674036, 0.9821574063216706, 0.3555078064299531],
        [0.9124490701578419, 0.9753266872784462, 0.3518533597970245],
        [0.9156931782520092, 0.9685354904254356, 0.34820569007261515],
        [0.9188613878011584, 0.9617532494963948, 0.34456629972324265],
        [0.9219411401959985, 0.9549849147932995, 0.3409370492608515],
        [0.9248899165263079, 0.948247019741531, 0.33731996777333784],
        [0.9278142496620382, 0.9414998557883345, 0.3337168327083648],
        [0.930591565440312, 0.9347905093445135, 0.33012998103815777],
        [0.9333375985220992, 0.928074834184405, 0.3265615469169363],
        [0.935971108343295, 0.921384812614979, 0.3230138415459749],
        [0.9385331275251968, 0.9147047487143539, 0.3194893313796646],
        [0.941028615969167, 0.9080328518084608, 0.3159906396320989],
        [0.943419864615293, 0.9013849121338212, 0.3125200216900484],
        [0.9457819798963852, 0.8947301765899248, 0.30908112084085126],
        [0.948014130728471, 0.8881110895508604, 0.30567519634043294],
        [0.9502464564071694, 0.8814728114137494, 0.3023077648025994],
        [0.9523487086523742, 0.8748712522095133, 0.29897881641458274],
        [0.9544348419615948, 0.8682572954460299, 0.2956947937778981],
        [0.956411752870728, 0.8616719638797333, 0.2924557053792813],
        [0.9583576301078739, 0.8550807238511935, 0.28926815137068224],
        [0.960210488502386, 0.8485116373841136, 0.2861325388392048],
        [0.9620231537308009, 0.8419407901977009, 0.28305544485779377],
        [0.9637524161478507, 0.8353882936883591, 0.28003762625495093],
        [0.9654377156641302, 0.8288358297838602, 0.2770858440538391],
        [0.967043069160827, 0.8223006128282059, 0.27420073280666335],
        [0.9686057099481874, 0.8157648637457888, 0.27138992938800466]
    ])
    # Add more color points from the MATLAB definition to match the full colormap
    # This is just a sample of the first few colors
    
    # Create and return the colormap
    return ListedColormap(thermal_colormap_data)

# Function to generate the 2D scattering plot
def generate_2d_scattering_plot(data, colormap, output_path, smoothing=False, colormap_name="speed"):
    """
    Generate a 2D scattering plot using the specified colormap.
    
    Parameters:
    -----------
    data : numpy.ndarray
        The scattering intensity data
    colormap : colormap
        Matplotlib colormap to use
    output_path : str
        Path to save the output image
    smoothing : bool, optional
        Whether to apply smoothing to the data
    colormap_name : str, optional
        Name of the colormap to use ("speed" or "thermal")
    """
    # Grid settings
    nq = 61
    ntheta = 61
    qmin_exponent = -2.1
    qmax_exponent = -0.9
    q_shift = 0.05
    
    # Ensure the data has the correct dimensions (nq x ntheta)
    if data.shape[0] > nq and data.shape[1] == ntheta:
        # Take a subset of rows to match nq
        data = data[:nq, :]
        print(f"Truncated data to shape: {data.shape}")
    elif data.shape[0] == ntheta and data.shape[1] > nq:
        # Transpose and take a subset if needed
        data = data.T[:nq, :]
        print(f"Transposed and truncated data to shape: {data.shape}")
    elif data.shape[0] != nq or data.shape[1] != ntheta:
        # If shape is completely different, resize the data
        # This is a simple approach - you might need a more sophisticated resizing method
        from scipy.ndimage import zoom
        zoom_factors = (nq / data.shape[0], ntheta / data.shape[1])
        data = zoom(data, zoom_factors, order=1)  # order=1 is linear interpolation
        print(f"Resized data to shape: {data.shape}")
    
    # If data needs smoothing
    if smoothing:
        smoothing_params = [2, 1]
        data_size = data.shape
        # Create periodic extension for theta dimension
        big_grid = np.hstack([data, data, data])
        # Apply moving average along theta dimension
        big_grid = uniform_filter1d(big_grid, size=smoothing_params[0], axis=1, mode='reflect')
        big_grid = big_grid[:, data_size[1]:(2*data_size[1])]
        # Apply moving average along q dimension
        data = uniform_filter1d(big_grid, size=smoothing_params[1], axis=0, mode='reflect')

    # Create qgrid and thetagrid for plotting
    qgrid = np.logspace(qmin_exponent, qmax_exponent, nq).reshape(-1, 1) * np.ones((1, ntheta))
    thetagrid = np.ones((nq, 1)) * np.linspace(0, np.pi, ntheta)

    # Convert data to base-10 for visualization
    # First replace any zeros or negative values with a small positive value to avoid log10 issues
    data_intensity = np.maximum(data, 1e-10)
    data_intensity = 10.0**data if not np.issubdtype(data.dtype, np.floating) else data_intensity

    # Transform data for visualization
    cmin, cmax = 2, 5  # Color limits
    plotx = (np.log10(qgrid) - qmin_exponent + q_shift) * np.cos(thetagrid)
    ploty = (np.log10(qgrid) - qmin_exponent + q_shift) * np.sin(thetagrid)
    plotz = np.clip(np.log10(data_intensity), a_min=cmin, a_max=cmax)

    # Setup plot with MATLAB-like formatting
    plt.figure(figsize=(8.7, 8.4))
    ax = plt.axes([0.15, 0.13, 0.8, 0.83])  # [left, bottom, width, height]

    # Plot positive and negative axes using contourf - follow MATLAB format exactly
    levels = np.arange(cmin, cmax + 0.01, 0.01)
    contourf_1 = ax.contourf(plotx, ploty, plotz, levels=levels, cmap=colormap, extend='both')
    contourf_2 = ax.contourf(-plotx, -ploty, plotz, levels=levels, cmap=colormap, extend='both')

    # Set axis properties
    ax.set_aspect('equal')
    ax.set_xlabel(r"$q_1 (\mathrm{\AA}^{-1})$", fontsize=35, fontname='Arial')
    ax.set_ylabel(r"$q_2 (\mathrm{\AA}^{-1})$", fontsize=35, fontname='Arial')
    
    # Calculate tick positions similar to MATLAB code
    q_incr = 0.5
    q_trunc = 0
    axislimit = qmax_exponent - qmin_exponent - q_trunc + q_shift
    
    # Generate ticks
    ticks = np.arange(q_shift, axislimit + q_incr/2, q_incr)  # Add q_incr/2 to avoid floating point issues
    ticks = np.concatenate([-np.flip(ticks), ticks])
    
    # Calculate tick labels values
    qmin_exponent_label = round(qmin_exponent / q_incr) * q_incr
    qmax_exponent_label = round(qmax_exponent / q_incr) * q_incr
    
    # The number of tick values should match the number of ticks
    n_positive_ticks = len(ticks) // 2  # Half of the ticks are positive
    ticklabels_vals = np.linspace(qmin_exponent_label, qmax_exponent_label, n_positive_ticks)
    
    # Create labels for each tick
    positive_labels = [f"$10^{{{int(val)}}}$" for val in ticklabels_vals]
    negative_labels = [f"$-10^{{{int(val)}}}$" for val in ticklabels_vals]
    ticklabels = negative_labels[::-1] + positive_labels
    
    # Set empty label for center (middle tick)
    if len(ticklabels) % 2 == 1:  # If odd number of labels
        middle_idx = len(ticklabels) // 2
        ticklabels[middle_idx] = ' '
    
    print(f"Number of ticks: {len(ticks)}, Number of labels: {len(ticklabels)}")
    
    # Set ticks and labels
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_yticklabels(ticklabels)
    
    ax.set_xlim([-axislimit, axislimit])
    ax.set_ylim([-axislimit, axislimit])
    
    # Set font size for tick labels
    ax.tick_params(axis='both', which='major', labelsize=30)
    
    # Turn off axes for a cleaner look
    ax.set_axis_off()
    
    # Save the figure with high resolution - use basename to include colormap name
    base_filename = os.path.splitext(output_path)[0]
    colormap_output_path = f"{base_filename}_{colormap_name}.png"
    plt.savefig(colormap_output_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    return colormap_output_path

def main():
    input_file_path = './output/test_scatteringprofiledatayz.txt'  
    output_file_path = './output/test_scatteringprofile.png'
    
    try:
        # Load data from input file ensuring it's numeric
        try:
            # Read the file content and split by lines
            with open(input_file_path, 'r') as f:
                lines = f.readlines()
            
            # Process each line to convert space-delimited strings to numeric arrays
            data_rows = []
            for line in lines:
                # Split the line by spaces and convert each element to float
                values = [float(val) for val in line.strip().split()]
                if values:  # Ensure we don't add empty rows
                    data_rows.append(values)
            
            # Convert the list of lists to a numpy array
            data = np.array(data_rows)
            
            print(f"Data shape: {data.shape}")
            
        except ValueError as e:
            print(f"Error parsing data: {str(e)}")
            # If parsing fails, try pandas with numeric conversion
            data = pd.read_csv(input_file_path, header=None, delim_whitespace=True, dtype=float).values
            print(f"Data shape after pandas read: {data.shape}")
        
        # Generate two plots using the custom colormaps
        # 1. Speed colormap
        try:
            # First try to load from MAT file
            colormap_file = 'speed_colormap.mat'
            if os.path.exists(colormap_file):
                mat_contents = loadmat(colormap_file)
                colormap_data = mat_contents.get('speed_colormap')
                speed_cmap = ListedColormap(colormap_data)
                print("Using speed colormap from MAT file")
            else:
                # Create custom speed colormap
                speed_cmap = create_speed_colormap()
                print("Using custom speed colormap")
                
            # Flip the colormap to match MATLAB version
            speed_cmap = speed_cmap.reversed()
            
            # Generate plot with speed colormap
            speed_output = generate_2d_scattering_plot(
                data=data, 
                colormap=speed_cmap, 
                output_path=output_file_path,
                colormap_name="speed"
            )
            print(f"Speed colormap plot saved at: {speed_output}")
        except Exception as e:
            print(f"Error with speed colormap: {str(e)}")
            # Fallback to cmocean
            speed_cmap = cmocean.cm.speed.reversed()
            speed_output = generate_2d_scattering_plot(
                data=data, 
                colormap=speed_cmap, 
                output_path=output_file_path,
                colormap_name="speed_cmocean"
            )
            print(f"Fallback speed colormap plot saved at: {speed_output}")
        
        # 2. Thermal colormap
        try:
            # First try to load from MAT file
            colormap_file = 'thermal_colormap.mat'
            if os.path.exists(colormap_file):
                mat_contents = loadmat(colormap_file)
                colormap_data = mat_contents.get('thermal_colormap')
                thermal_cmap = ListedColormap(colormap_data)
                print("Using thermal colormap from MAT file")
            else:
                # Create custom thermal colormap
                thermal_cmap = create_thermal_colormap()
                print("Using custom thermal colormap")
                
            # Flip the colormap to match MATLAB version
            thermal_cmap = thermal_cmap.reversed()
            
            # Generate plot with thermal colormap
            thermal_output = generate_2d_scattering_plot(
                data=data, 
                colormap=thermal_cmap, 
                output_path=output_file_path,
                colormap_name="thermal"
            )
            print(f"Thermal colormap plot saved at: {thermal_output}")
        except Exception as e:
            print(f"Error with thermal colormap: {str(e)}")
            # Fallback to cmocean
            if hasattr(cmocean.cm, 'thermal'):
                thermal_cmap = cmocean.cm.thermal.reversed()
                thermal_output = generate_2d_scattering_plot(
                    data=data, 
                    colormap=thermal_cmap, 
                    output_path=output_file_path,
                    colormap_name="thermal_cmocean"
                )
                print(f"Fallback thermal colormap plot saved at: {thermal_output}")
            else:
                print("Thermal colormap not available in cmocean")
    
    except FileNotFoundError:
        print(f"Error: File not found at {input_file_path}. Please check the file path.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
