# Shallow Recurrent Decoder (SHRED) Application in Time-Series Prediction
### Huy Huynh

## Abstract
SHRED network architecture leverages a shallow decoder network to reconstruct high-dimensional temporal state from trajectory of sensor measurements of the state then pass through a recurrent layer (LSTM). Some application of SHRED is predicting the state of the whole system given only parts of that system measurements.

## Introduction
The NOAA Optimum Interpolation SST V2 [dataset](https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.html) is used to test SHRED performance, which give the weekly mean sea-surface temperature. There are three parameters that can be changed with the SST dataset: the number of sensor location randomly chosen to use, trajectory length (lags) and percentage of Gaussian noise added to the data. This page will analyze the effects of these three parameter on the performance of SHRED in prediction.

## Background
The SHRED architecture can be written as:

<p align="center">
  $$H(\{y_{i}\}^{t}_{i=t-k}) = F(G(\{y_{i}\}^{t}_{i=t-k});W_{RN});W_{SD})$$
</p>

where $\mathcal F$ is a feed forward network parameterized by weights $W_{SD}$, $\mathcal G$ is a LSTM network parameterized by weights $W_{RN}$, and $\{ y_i \} _{i=t-k}^t$ is a trajectory of sensor measurements of a high-dimensional spatio-temporal field $\{ x_i \} _{i=t-k}^t$.

## Implementation
The SHRED model used is from this Github [repository](https://github.com/Jan-Williams/pyshred) by Jan P. Williams, Olivia Zahn, and J. Nathan Kutz. There are some modifications to the original repository in order to add some visualization to the output and loading the full SST data based on a [fork](https://github.com/shervinsahba/pyshred) by Shervinsahba. The summary of the SHRED network is shown in the figure below.

<p align="center">
  <img src="https://github.com/hhuynh000/EE399_HW6/blob/main/resources/SHRED.png" width="500"/>
</p>
<p align="center">
  Figure 1. SHRED Model Architecture
</p>

In addition, a Gaussian noise with a mean of 0 and standard deviation of 1 parameter is added to the data pre-proccessing for the SHRED model. The code implementation is shown in the figure below.

<p align="center">
  <img src="https://github.com/hhuynh000/EE399_HW6/blob/main/resources/Noise.png" width="500"/>
</p>
<p align="center">
  Figure 2. Gaussian Noise Implementation
</p>

For all training, the parameters is used:
  - Batch size = 64
  - Number of epochs = 1000
  - Learning rate = 0.003

## Results
### Base test
Using the parameters of number of sensors = 3, lags = 52 and noise = 0, the model after training has a mean square error of 0.01979 between the truth and reconstructed weekly mean sea-surface temperature. One of the sampled reconstructed output compared with the ground truth is shown in the figure below.

<p align="center">
  <img src="https://github.com/hhuynh000/EE399_HW6/blob/main/resources/recons.png" width="500"/>
</p>
<p align="center">
  Figure 3. Ground Truth vs. Reconstruction
</p>

### Time lags test
Using the fixed parameters of number of sensors = 3, noise = 0, while varying the time lags by 13, 26, 39, 52 representing the first, second, third and fourth quarter of the year respectively. The plot of the mean square error as a function of time lags is shown in the figure below.

<p align="center">
  <img src="https://github.com/hhuynh000/EE399_HW6/blob/main/resources/lags.png" width="500"/>
</p>
<p align="center">
  Figure 4. MSE vs. Time Lags
</p>

### Gaussian noise test
Using the fixed parameters of number of sensors = 3, lags = 52, while varying the noise by 0, 0.25, 0.5, 0.75 and 1. The plot of the mean square error as a function of Gaussian noise is shown in the figure below.

<p align="center">
  <img src="https://github.com/hhuynh000/EE399_HW6/blob/main/resources/gaussian.png" width="500"/>
</p>
<p align="center">
  Figure 5. MSE vs. Gaussian Noise
</p>

### Gaussian noise test
Using the fixed parameters of number of sensors = 3, lags = 52, while varying the noise by 0, 0.25, 0.5, 0.75 and 1. The plot of the mean square error as a function of Gaussian noise is shown in the figure below.

<p align="center">
  <img src="https://github.com/hhuynh000/EE399_HW6/blob/main/resources/sensors.png" width="500"/>
</p>
<p align="center">
  Figure 6. MSE vs. Gaussian Noise
</p>

## Conclusion
