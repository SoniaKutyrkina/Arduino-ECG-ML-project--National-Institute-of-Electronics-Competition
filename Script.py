#-----IMPORTING REQUIRED LIBRARIES----#

#Data manipulation
import pandas as pd 
import numpy as np

#Signal Processing
import pywt #wavelet transforms
import scipy.signal #signal processing utilities
from scipy.signal import medfilt, butter, filtfilt, iirnotch #specific filtering functions

#Machine learning preprocessing, model selection, and evaluation
from sklearn.preprocessing import MinMaxScaler #Normalising Data
from sklearn.model_selection import train_test_split #Splitting data into training and testing
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report, roc_auc_score
) #Evaluation metrics

#Interactive plotting
import plotly.graph_objs as go
import plotly.express as px

# Deep Learning (Keras)
from keras.models import Sequential #For building sequential models
from keras.utils import plot_model #visualise model structure
from keras.layers import (
    LSTM, Bidirectional, Dense, Reshape,
    Conv1D, MaxPooling1D, Dropout, Flatten
) #layers for CNN + LSTM model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau #training callbacks
from keras.optimizers import Adam #optimizer



#-----IMPORTING THE DATASET-----#

#Load the ECG dataset from TensorFlow hosted CSV
df = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header = None)



#----RUNNING DATABASE CHECKS----#

head = df.head()
#displays the first 5 rows of data
shape = df.shape
#displays the shape of the dataset
info = df.info()
#displays the dataframe information
columns = df.columns
#displays the column names 
target_var = df[140].unique()
#display the values of the target variable, the last column of the dataset -> index 140
num_aberrantandnormal = df[140].value_counts()
#display the number of examples for aberrant and normal ECGs, 1 is normal & 0 is abnormal ecg
column_datype = df.dtypes
#displaying the datatypes of each column
missing_val = df.isna().sum().sum()
#checking missing values for the entire dataset

#-----PLOTTING NORMAL AND ABERRANT ECG SIGNALS-----#

#Selects the first 10 abnormal and normal ECG signals 
abnormal = df[df.loc[:,140]==0][:10]
normal = df[df.loc[:,140]==1][:10]

fig = go.Figure() #initialialise plotly figure

#Legend visibility setup
leg = [False] * abnormal.shape[0]
leg[0] = True

#Plot abnormal ECG signals in red
for i in range(abnormal.shape[0]):
    fig.add_trace(go.Scatter(x=np.arange(abnormal.shape[1]),y=abnormal.iloc[i,:], name="Abnormal ECG", mode='lines', 
    marker_color='rgba(255,0,0,0.9)', showlegend=leg[i]))

#Plot normal ECG signals in green
for j in range(normal.shape[0]):
    fig.add_trace(go.Scatter(x=np.arange(normal.shape[1]),y=normal.iloc[j,:], name="Normal ECG", mode='lines', 
    marker_color='rgba(0,255,0,1)', showlegend=leg[j]))

#set layout
fig.update_layout(xaxis_title="time (ms)", yaxis_title="Signal", title = {'text':'Normal vs Abnormal ECG signals',
'xanchor':'center','yanchor':'top','x':0.5},bargap=0)
fig.update_traces(opacity=0.5)
fig.show()

#-----DATA PREPROCESSING-----#

#split dataset into features and labels 
ecg_data = df.iloc[:,:-1]
labels = df.iloc[:,-1]

#Normalising the dataset
scaler = MinMaxScaler(feature_range=(-1,1))
ecg_data = scaler.fit_transform(ecg_data)

#filter the dataset testing 4 techniques
#Median Filtering 
ecg_medfilt = medfilt(ecg_data, kernel_size=3)

#Low-pass filtering
lowcut = 0.05
highcut = 20.0
nyquist = 0.5 * 360.0
low = lowcut / nyquist
high = highcut / nyquist 
b,a = butter(4, [low, high], btype='band')
ecg_lowpass = filtfilt(b, a, ecg_data)

#Wavelet filtering
coeffs = pywt.wavedec(ecg_data, 'db4', level=1)
threshold = np.std(coeffs[-1]) * np.sqrt(2*np.log(len(ecg_data)))
coeffs[1:]=(pywt.threshold(i, value=threshold, mode='soft') for i in coeffs[1:])
ecg_wavelet = pywt.waverec(coeffs, 'db4')

#Band-stop filtering
notch_freq = 50.0  # Hz
quality_factor = 30.0
w0 = notch_freq / (nyquist * 2)  # Normalized Frequency
b_notch, a_notch = iirnotch(w0, quality_factor)
ecg_notch = filtfilt(b_notch, a_notch, ecg_data, axis=0)


#Plotting graphs of unfiltered and filtered signal
#original signal
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(ecg_data.shape[0]), y=ecg_data[30], mode='lines', name='Original ECG signal'))

#filtered ecg signal
fig.add_trace(go.Scatter(x=np.arange(ecg_medfilt.shape[0]), y=ecg_medfilt[30], mode='lines', name='Median filtered ECG signal'))
fig.add_trace(go.Scatter(x=np.arange(ecg_lowpass.shape[0]), y=ecg_lowpass[30], mode='lines', name='Low-pass filtered ECG signal'))
fig.add_trace(go.Scatter(x=np.arange(ecg_wavelet.shape[0]), y=ecg_wavelet[30], mode='lines', name='Wavelet filtered ECG signal'))
fig.add_trace(go.Scatter(x=np.arange(ecg_notch.shape[1]), y=ecg_notch[30], mode='lines', name='Notch filtered'))
fig.show()

#determining best filtering technique 
#pad the signal with zeros
def pad_data(original_data, filtered_data):
    diff = original_data.shape[1] - filtered_data.shape[1]
    if diff > 0:
        padding = np.zeros((filtered_data.shape[0], original_data.shape[1]))
        padded_data = np.concentrate((filtered_data, padding))
    elif diff < 0:
        padded_data = filtered_data[:,:-abs(diff)]
    elif diff == 0:
        padded_data = filtered_data
    return padded_data 

#compute mean squared error 
def mse(original_data, filtered_data):
    filter_data = pad_data(original_data, filtered_data)
    return np.mean((original_data - filter_data) ** 2)

mse_value_m= mse(ecg_data, ecg_medfilt)
mse_value_l= mse(ecg_data, ecg_lowpass)
mse_value_w= mse(ecg_data, ecg_wavelet)
mse_value_n = mse(ecg_data, ecg_notch)


print("MSE value of Median Filtering:", mse_value_m)
print("MSE value of Low-pass Filtering:", mse_value_l)
print("MSE value of Wavelet Filtering:", mse_value_w)
print("MSE value of band-stop Filtering:", mse_value_n)

#hybrid filtering approach: wavelet filtering + median filtering
hybrid = medfilt(ecg_wavelet, kernel_size=3)
 
#-----SPLITTING DATA INTO TRAIN AND TEST----#

X_train, X_test, y_train, y_test = train_test_split(hybrid, labels, test_size = 0.2, random_state = 42)

#-----FEATURE EXTRACTION-----#

#Feature extraction of the train set 
features = []
for i in range(X_train.shape[0]):
    r_peaks = scipy.signal.find_peaks(X_train[i])[0]
    r_amplitudes = []
    t_amplitudes = []
    for r_peak in r_peaks:
        t_peak = np.argmin(X_train[i][r_peak:r_peak + 200] + r_peak)
        r_amplitudes.append(X_train[i][r_peak])
        t_amplitudes.append(X_train[i][t_peak])

    std_r_amp = np.std(r_amplitudes)
    mean_r_amp = np.mean(r_amplitudes)
    median_r_amp = np.median(r_amplitudes)
    sum_r_amp = np.sum(r_amplitudes)

    std_t_amp = np.std(t_amplitudes)
    mean_t_amp = np.mean(t_amplitudes)
    median_t_amp = np.median(t_amplitudes)
    sum_t_amp = np.sum(t_amplitudes)

    rr_intervals = np.diff(r_peaks)
    time_duration = (len(X_train[i])-1) / 1000
    sampling_rate = len(X_train[i]) / time_duration
    duration = len(X_train[i]) / sampling_rate 
    heart_rate = (len(r_peaks) / duration ) * 60
    qrs_duration = []
    for j in range(len(r_peaks)):
        qrs_duration.append(r_peaks[j]-r_peaks[j-1])
    
    std_qrs = np.std(qrs_duration)
    mean_qrs = np.mean(qrs_duration)
    median_qrs = np.median(qrs_duration)
    sum_qrs = np.sum(qrs_duration)

    std_rr = np.std(rr_intervals)
    mean_rr = np.mean(rr_intervals)
    median_rr = np.median(rr_intervals)
    sum_rr = np.sum(rr_intervals)

    std = np.std(X_train[i])
    mean = np.mean(X_train[i])
    features.append([mean, std, std_qrs, mean_qrs, median_qrs, sum_qrs, std_r_amp, mean_r_amp, median_r_amp, 
    sum_r_amp, std_t_amp, mean_t_amp, median_t_amp, sum_t_amp, sum_rr, std_rr, mean_rr, median_rr, heart_rate])
features = np.array(features)


#feature extraction of the test set
X_test_fe = []
for i in range(X_test.shape[0]):
    r_peaks = scipy.signal.find_peaks(X_test[i])[0]
    r_amplitudes = []
    t_amplitudes = []
    for r_peak in r_peaks:
        t_peak = np.argmin(X_test[i][r_peak:r_peak + 200] + r_peak)
        r_amplitudes.append(X_test[i][r_peak])
        t_amplitudes.append(X_test[i][t_peak])

    std_r_amp = np.std(r_amplitudes)
    mean_r_amp = np.mean(r_amplitudes)
    median_r_amp = np.median(r_amplitudes)
    sum_r_amp = np.sum(r_amplitudes)

    std_t_amp = np.std(t_amplitudes)
    mean_t_amp = np.mean(t_amplitudes)
    median_t_amp = np.median(t_amplitudes)
    sum_t_amp = np.sum(t_amplitudes)

    rr_intervals = np.diff(r_peaks)
    time_duration = (len(X_test[i])-1) / 1000
    sampling_rate = len(X_test[i]) / time_duration
    duration = len(X_test[i]) / sampling_rate 
    heart_rate = (len(r_peaks) / duration ) * 60
    qrs_duration = []
    for j in range(len(r_peaks)):
        qrs_duration.append(r_peaks[j]-r_peaks[j-1])
    
    std_qrs = np.std(qrs_duration)
    mean_qrs = np.mean(qrs_duration)
    median_qrs = np.median(qrs_duration)
    sum_qrs = np.sum(qrs_duration)

    std_rr = np.std(rr_intervals)
    mean_rr = np.mean(rr_intervals)
    median_rr = np.median(rr_intervals)
    sum_rr = np.sum(rr_intervals)

    std = np.std(X_test[i])
    mean = np.mean(X_test[i])
    X_test_fe.append([mean, std, std_qrs, mean_qrs, median_qrs, sum_qrs, std_r_amp, mean_r_amp, median_r_amp,
     sum_r_amp, std_t_amp, mean_t_amp, median_t_amp, sum_t_amp, sum_rr, std_rr, mean_rr, median_rr, heart_rate])
X_test_fe = np.array(X_test_fe)



#-----MODEL BUILDING AND TRAINING----#

num_features = features.shape[1]
features = np.asarray(features).astype('float32')
features = features.reshape(features.shape[0], features.shape[1], 1)
X_test_fe = X_test_fe.reshape(X_test_fe.shape[0], X_test_fe.shape[1],1)

#CNN+LSTM
model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(num_features, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.3))

model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Optimizer and callbacks
opt = Adam(learning_rate=0.001)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Compile the model
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
 
# Save the model
model.save("ecg_model.h5")

# Visualise the model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Train the model
history = model.fit(
    features, y_train,
    validation_data=(X_test_fe, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[reduce_lr, early_stop]
)

# Predict
y_pred = model.predict(X_test_fe)
y_pred = [1 if p > 0.5 else 0 for p in y_pred]
X_test_fe = np.asarray(X_test_fe).astype('float32')

#-----MODEL EVALUATION-----#

#calculating metrics
acc = accuracy_score(y_test, y_pred)
auc = round(roc_auc_score(y_test, y_pred),2)
all_met = classification_report(y_test, y_pred)

#displaing metrics
print("Accuracy: ", acc*100, "%")
print("\n")
print("AUC:", auc)
print("\n")
print("Classification Report: \n", all_met)
print("\n")

#----CALCULATING AND DISPLAYING THE CONFUSION MATRIX-----#

conf_mat = confusion_matrix(y_test, y_pred)
conf_mat_df = pd.DataFrame(conf_mat, columns = ["Predicted Negative", 'Predicted Positive'], index= ['Actual Negative','Actual Positive'])
fig = px.imshow(conf_mat_df,text_auto=True, color_continuous_scale='Blues')
fig.update_yaxes(side='top',title_text='Predicted')
fig.update_yaxes(title_text='Actual')
fig.show()

#-----PLOTTING THE TRAINING AND VALIDATION ERROR-----#

fig = go.Figure()
fig.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Training'))
fig.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Validation'))
fig.update_layout(xaxis_title="Epoch", yaxis_title="Error", title= {'text': 'Model Error', 'xanchor':'center', 'yanchor':'top', 'x':0.5}, bargap=0)
fig.show()