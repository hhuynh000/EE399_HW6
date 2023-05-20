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
