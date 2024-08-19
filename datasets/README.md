# Datasets

## Important Notice
For using the two datasets METR-LA and PEMS-BAY, please unzip the file named “data_in_12_out_12_rescale_True.zip” in the two dataset dictionary first.

## Datasets Description

### 1. METR-LA

**Source**: [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting, ICLR'18](https://github.com/liyaguang/DCRNN). [Data Link](https://github.com/liyaguang/DCRNN).

**Description**: METR-LA is a traffic speed dataset collected from loop-detectors located on the LA County road network. It contains data of 207 sensors over a period of 4 months from Mar 2012 to Jun 2012. The traffic information is recorded at the rate of every 5 minutes. METR-LA also includes a sensor graph to indicate dependencies between sensors. DCRNN computes the pairwise road network distances between sensors and build the adjacency matrix using a thresholded Gaussian kernel. Details can be found in the [paper](https://arxiv.org/pdf/1707.01926.pdf).

**Period**: 2012/3/1 0:00:00 -> 2012/6/27 23:55:00

**Number of Time Steps**: 34272

**Dataset Split**: 7:1:2.

**Variates**: Each variate represents the traffic speed of a sensor.

**Typical Settings**:

- Spatial temporal forecasting.

### 2. PEMS-BAY

**Source**: [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting, ICLR'18](https://github.com/liyaguang/DCRNN). [Data Link](https://github.com/liyaguang/DCRNN).

**Description**: PEMS-BAY is a traffic speed dataset collected from California Transportation Agencies (CalTrans) Performance Measurement System (PeMS). It contains data of 325 sensors in the Bay Area over a period of 6 months from Jan 2017 to June 2017. The traffic information is recorded at the rate of every 5 minutes. PEMS-BAY also includes a sensor graph to indicate dependencies between sensors. DCRNN computes the pairwise road network distances between sensors and build the adjacency matrix using a thresholded Gaussian kernel. Details can be found in the [paper](https://arxiv.org/pdf/1707.01926.pdf).

**Period**: 2017/1/1 0:00:00 -> 2017/6/30 23:55:00

**Number of Time Steps**: 52116

**Dataset Split**: 7:1:2.

**Variates**: Each variate represents the traffic speed of a sensor.

**Typical Settings**:

-  Spatial temporal forecasting.
