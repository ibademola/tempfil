import numpy as np
from scipy.optimize import curve_fit
from datetime import datetime
from .gdal_utils import get_atc_array, m_window
from osgeo import gdal

def get_acquisition_date(file_name):
    date_str = file_name.split('_')[3]  # Extracting the date part from the filename
    acquisition_date = datetime.strptime(date_str, '%Y%m%d').date()
    return acquisition_date

def annual_temperature_cycle(x, A, B, C, D):
    return A * np.sin(2 * np.pi * x / 365.25 + B) + C * x + D

def get_best_fit_parameters_and_lst(date, pixel_values_dict):
    acquisition_dates = []
    acquisition_dates_doy = []
    pixel_values = []

    for date_str, value in pixel_values_dict.items():
        if value >= 265 and value <= 320:
            acquisition_dates.append(date_str)
            acquisition_dates_doy.append(datetime.strptime(date_str, '%Y-%m-%d').timetuple().tm_yday)
            pixel_values.append(value)

    popt, pcov = curve_fit(annual_temperature_cycle, acquisition_dates_doy, pixel_values, bounds=([-np.inf, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]))

    x = datetime.strptime(date, '%Y-%m-%d').timetuple().tm_yday
    lst_value = annual_temperature_cycle(x, *popt)

    return lst_value

def find_missing_pixels(input_file_path, window_radius):
    # Open the input TIFF file to identify missing pixels
    input_dataset = gdal.Open(input_file_path)
    input_array = input_dataset.GetRasterBand(1).ReadAsArray()
    missing_pixels = set()
    surrounding_window = set()

    rows, cols = input_array.shape

    # Calculate bounds for the surrounding window
    min_i = 0
    max_i = rows
    min_j = 0
    max_j = cols

    # Iterate through the input array
    for i in range(rows):
        for j in range(cols):
            # Check if the pixel is missing
            if input_array[i, j] < 12 or input_array[i, j] > 400:
                missing_pixels.add((i, j))

                # Find surrounding window
                for x in range(max(0, i - window_radius), min(rows, i + window_radius + 1)):
                    for y in range(max(0, j - window_radius), min(cols, j + window_radius + 1)):
                        surrounding_window.add((x, y))

    # Convert set to list with 2D array representation
    surrounding_window = [[index[0], index[1]] for index in surrounding_window]

    return surrounding_window

def get_atc_array(input_file_path, folder_path, date, window_radius):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.tif'):  # Assuming all Landsat files are TIFFs
            file_path = os.path.join(folder_path, file_name)
            dataset = gdal.Open(file_path)
            rows = dataset.RasterYSize
            cols = dataset.RasterXSize
            break  # We assume all TIFF files have the same dimensions

    atc_array = np.zeros((rows, cols), dtype=float)  # 5 for A, B, C, D, and LST

    #Get index of missing missing pixel including surrounding window
    missing_pixel = find_missing_pixels(input_file_path, window_radius)
    for index in missing_pixel:
        x, y = index  # Get row and column indices from missing_pixel
        
        # initialize directory to store pixel (x,y) values on different acusition days
        pixel_values_dict = {}

        for file_name in os.listdir(folder_path):
            if file_name.endswith('.tif'):  # Assuming all Landsat files are TIFFs
                file_path = os.path.join(folder_path, file_name)
                acquisition_date = get_acquisition_date(file_name)
                pixel_value = extract_pixel_value(file_path, x, y)
                pixel_values_dict[str(acquisition_date)] = pixel_value

        atc_array[x, y] = get_best_fit_parameters_and_lst(date, pixel_values_dict)

    return atc_array

def rec(ts, atc):
    wx, wy = ts.shape[0], ts.shape[1]
    cp_x, cp_y = wx // 2, wy // 2
    
    if np.isnan(ts[cp_x, cp_y]):
        ls = np.abs(atc[cp_x, cp_y] - atc)
        sd_atc = np.std(atc)
        rs = 2 * sd_atc / 4
        r, c = np.where(ls <= rs)
        sp = np.stack((r, c), axis=1)
        msp = sp
        for m in range(len(msp) - 1, -1, -1):
            if np.isnan(ts[msp[m][0], msp[m][1]]):
                msp = np.delete(msp, m, axis=0)
        D1s = np.zeros(len(msp))
        D2s = np.zeros(len(msp))
        ws = np.zeros(len(msp))
        for t in range(len(msp)):
            D1s[t] = 1 + 2 * (np.sqrt((msp[t, 0] - wy)*2 + (msp[t, 1] - wx)*2)) / 4
            D2s[t] = np.abs(atc[msp[t, 0], msp[t, 1]] - atc[cp_x, cp_y])
            if D2s[t] == 0:
                continue
            ws[t] = D1s[t] * np.log(1 + D2s[t])
            ws[t] = 1 / ws[t]
        weight_sum = np.sum(ws)
        if weight_sum != 0:
            weight = ws / weight_sum
            rts = atc[cp_x, cp_y]
            for m in range(len(msp)):
                if not np.isnan(ts[msp[m][0], msp[m][1]]):
                    right = weight[m] * (ts[msp[m][0], msp[m][1]] - atc[msp[m][0], msp[m][1]])
                    rts += right
        else:
            rts = atc[cp_x, cp_y]
    else:
        rts = ts[cp_x, cp_y]
    return rts

def m_window(ts, atc, windowsize=10, stride=1):

    padimage_1 = np.pad(ts, windowsize, 'reflect')
    padimage_2 = np.pad(atc, windowsize, 'reflect')
    output = np.zeros_like(ts)

    for i in range(ts.shape[0]):
        for j in range(ts.shape[1]):
            indx_i = i + windowsize
            indx_j = j + windowsize
            if indx_i + windowsize + 1 <= padimage_1.shape[0] and indx_j + windowsize + 1 <= padimage_1.shape[1]:
                output[i, j] = rec(padimage_1[indx_i - windowsize:indx_i + windowsize + 1, indx_j - windowsize:indx_j + windowsize + 1],
                                   padimage_2[indx_i - windowsize:indx_i + windowsize + 1, indx_j - windowsize:indx_j + windowsize + 1])

    return output 

def process_data(input_file_path, folder_path, date, window_radius):
    atc_01_06 = get_atc_array(input_file_path, folder_path, date, window_radius)

    dataset = gdal.Open(input_file_path)
    Ts_f = dataset.GetRasterBand(1).ReadAsArray()
    Ts_f[(Ts_f < 200) | (Ts_f > 400)] = np.nan

    F_01_06 = m_window(Ts_f, atc_01_06, windowsize=10)

    return F_01_06