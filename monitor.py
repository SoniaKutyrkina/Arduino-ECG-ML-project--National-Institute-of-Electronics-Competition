#-----IMPORT LIBRARIES-----#
import serial  # For communicating with the ECG device via serial port
import matplotlib.pyplot as plt  # For plotting ECG signals
import matplotlib.animation as animation  # For live updating of the plot
from collections import deque  # Efficient fixed-length queue to store incoming data
import numpy as np  # Numerical operations
import pywt  # Wavelet transforms for signal denoising
from scipy.signal import medfilt, butter, filtfilt, iirnotch, find_peaks  # Filtering and peak detection
from sklearn.preprocessing import MinMaxScaler  # Normalization of input
from keras.models import load_model  # Load the pre-trained CNN+LSTM model

#-----SERIAL PORT SETTINGS-----#
BAUD = 9600  # Baud rate of the serial connection
MAX_POINTS = 360  # Number of points per ECG heartbeat (~1 second at 360Hz)
PORT = "/dev/cu.usbserial-A5069RR4"  
ser = serial.Serial(PORT, BAUD)  # Initialize serial communication with device

#-----REAL-TIME PLOTTING SETUP-----#
data = deque([0]*MAX_POINTS, maxlen=MAX_POINTS)  # Fixed-length queue to store ECG points
fig, ax = plt.subplots()  # Create a matplotlib figure
line, = ax.plot(data)  # Plot initial line for ECG
ax.set_ylim(0, 1023)  # Set Y-axis limits to match ADC output (10-bit resolution)

#-----LOAD TRAINED MODEL-----#
model = load_model("ecg_model.h5")  # Load previously trained CNN+LSTM model

#-----DATA PREPROCESSING FUNCTIONS-----#
def preprocess_ecg(raw_signal):
    """
    Normalize, filter, and extract features from raw ECG for ML model.
    Input: raw_signal -> list or array of raw ECG values
    Output: preprocessed features shaped for CNN+LSTM
    """
    raw_signal = np.array(raw_signal).reshape(1, -1)  # Convert to numpy array and reshape for scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))  # Initialize MinMaxScaler
    normalized = scaler.fit_transform(raw_signal)  # Normalize ECG between -1 and 1

    # Wavelet filtering
    coeffs = pywt.wavedec(normalized, 'db4', level=1)  # Decompose signal
    threshold = np.std(coeffs[-1]) * np.sqrt(2*np.log(len(normalized[0])))  # Universal threshold
    coeffs[1:] = (pywt.threshold(i, value=threshold, mode='soft') for i in coeffs[1:])  # Denoising
    wavelet_filtered = pywt.waverec(coeffs, 'db4')  # Reconstruct signal

    # Median filtering
    hybrid = medfilt(wavelet_filtered, kernel_size=3)  # Remove small spikes

    # Feature extraction
    return extract_features(hybrid)  # Return features for the ML model

def extract_features(X):
    """
    Extract features like R/T peaks, RR intervals, QRS duration, heart rate, and signal stats.
    Input: 1D ECG array
    Output: reshaped features ready for CNN+LSTM model
    """
    features_list = []
    for i in range(X.shape[0]):
        r_peaks = np.array(find_peaks(X[i])[0])  # Detect R-peaks
        if len(r_peaks) < 2: r_peaks = np.array([0, 1])  # Fallback for empty peaks
        r_amplitudes, t_amplitudes = [], []
        for r_peak in r_peaks:
            t_peak = np.argmin(X[i][r_peak:r_peak + 200] + r_peak)  # Find T-wave minima after R-peak
            r_amplitudes.append(X[i][r_peak])
            t_amplitudes.append(X[i][t_peak])

        # R-wave statistics
        std_r_amp = np.std(r_amplitudes)
        mean_r_amp = np.mean(r_amplitudes)
        median_r_amp = np.median(r_amplitudes)
        sum_r_amp = np.sum(r_amplitudes)

        # T-wave statistics
        std_t_amp = np.std(t_amplitudes)
        mean_t_amp = np.mean(t_amplitudes)
        median_t_amp = np.median(t_amplitudes)
        sum_t_amp = np.sum(t_amplitudes)

        # RR intervals
        rr_intervals = np.diff(r_peaks) if len(r_peaks) > 1 else np.array([1])
        std_rr, mean_rr, median_rr, sum_rr = np.std(rr_intervals), np.mean(rr_intervals), np.median(rr_intervals), np.sum(rr_intervals)

        # QRS duration
        qrs_duration = [r_peaks[j]-r_peaks[j-1] for j in range(1,len(r_peaks))] if len(r_peaks)>1 else [1]
        std_qrs, mean_qrs, median_qrs, sum_qrs = np.std(qrs_duration), np.mean(qrs_duration), np.median(qrs_duration), np.sum(qrs_duration)

        # Heart rate
        duration = len(X[i]) / 360.0  # Sampling rate 360Hz
        heart_rate = (len(r_peaks)/duration) * 60  # BPM

        # Signal statistics
        std, mean = np.std(X[i]), np.mean(X[i])

        # Append all features
        features_list.append([mean, std, std_qrs, mean_qrs, median_qrs, sum_qrs,
                              std_r_amp, mean_r_amp, median_r_amp, sum_r_amp,
                              std_t_amp, mean_t_amp, median_t_amp, sum_t_amp,
                              sum_rr, std_rr, mean_rr, median_rr, heart_rate])
    return np.array(features_list).reshape(1, -1, 1)  # Reshape for CNN+LSTM

#-----UPDATE FUNCTION FOR ANIMATION-----#
def update(frame):
    """
    Called every frame by FuncAnimation:
    1. Reads new data from serial port
    2. Updates the plot
    3. Runs ML prediction if enough data collected
    """
    if ser.in_waiting:  # Check if data is available
        raw = ser.readline().decode().strip()  # Read and decode line
        try:
            val = int(raw)  # Convert to integer
            data.append(val)  # Append to sliding window
        except:
            pass  # Ignore non-integer data

        # Update the plot
        line.set_ydata(data)
        line.set_xdata(range(len(data)))

        # Predict if enough points collected
        if len(data) == MAX_POINTS:
            processed = preprocess_ecg(list(data))  # Preprocess the ECG
            pred = model.predict(processed)  # Run prediction
            label = "Normal" if pred[0][0] > 0.5 else "Abnormal"
            print("ECG Prediction:", label)  # Print result

    return line,

#-----RUN LIVE ANIMATION-----#
ani = animation.FuncAnimation(fig, update, interval=20)  # Call update every 20ms
plt.show()  # Show live plot