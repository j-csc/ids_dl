# IDS/IPS with Deep learning - A Keras implementation of a proof-of-concept Intrusion Detection and Prevention system

This repository contains the following: 

1. A python script that downloads the raw csv files needed for sampling.

> To download the data from ISCX, make sure you have python 3.x installed and run `python3 downloadData.py`. Then, wait until it prompts that the download is complete. The total file size is around 887 mb and would take rather long...

2. A Jupyter notebook that contains my proof of concept code for a deep learning framework that detects intrusions. It also contains everything from data wrangling to creating baseline models such as KMeans and an Autoencoder network.

> To run the notebook above, have Anaconda installed and run `jupyter notebook POC.ipynb` in the downloaded repository.s