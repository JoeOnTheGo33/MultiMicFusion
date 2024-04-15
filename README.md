# Multi-Microphone Audio Data Fusion
**Authors:** Gary Du , Alexander Gruhl, Abhi Gupta, Josiah Taylor, Joseph Younger

**Course:** MTE 546, University of Waterloo

## Abstract

The project for this paper was to determine the effectiveness of multi-microphone audio data fusion. This was done by recording a conversation and having multiple microphones around it. Fusion algorithms which combined all the recordings were applied to the recordings. The fused and filtered recording was compared to the unfiltered recordings using different comparison metrics. Through multiple results finding methods such as histograms, spectrogram sound entropy, and human perception tests, it was determined that the fused recording had significantly less background noise and more distinct information than the unfiltered recordings.

**Index Terms** — Audio, Data Fusion, Noise Reduction, Moving Average, and Signal Entropy

## Description

Audio recordings is stored in `data'. It includes 3 surrounding sensors and one central sensor.

The main code is in [fusion/main.ipynb](fusion/main.ipynb).

An implemeentation of moving average alone is in `fusion_moving_average`.

## Sensor Info
Sensor 3 - Josiah's Laptop
Sensor 2 - Alex's Phone
Sensor 1 - Joseph's Phone
