## Overview
Welcome to the repository for the **"Spatial encoding of BOLD fMRI time series for categorizing static images across visual datasets: A pilot study on human vision"**

Functional MRI (fMRI) is a powerful tool for studying brain functionality by detecting changes in oxygenated blood flow associated with brain activity. In this study, we leverage the publicly available BOLD5000 dataset, which contains fMRI scans captured while subjects viewed 5254 images from diverse categories. These images are drawn from three standard computer vision datasets: COCO, ImageNet, and SUN.

## Methodology
*To analyze vision-related neural activity, we perform spatial encoding on fMRI BOLD time series data. Specifically, we use two classical methods, Gramian Angular Field (GAF) and Markov Transition Field (MTF), to transform the fMRI time series into 2D representations. These 2D representations correspond to images from the COCO, ImageNet, and SUN datasets.
For image classification, we employ Convolutional Neural Networks (CNNs). Initially, individual GAF and MTF features are fed into separate CNN models for binary classification. Later, we introduce a parallel CNN architecture that combines the 2D features obtained from GAF and MTF. This parallel CNN model is designed to classify images across the COCO, ImageNet, and SUN datasets.*
## Training
To initiate training and analysis:

 Ensure that the required dependencies are installed by executing:

```bash
   pip install -r requirements.txt
```
## Repository Structure
   
   `tencon/`: Contains Python source code files.
   
   `model_preprocessing.py`:Contains data preprocessing functions and training CNN models functions.

   `README.md`: This file, offering an overview of the project.
   
   `requirements.txt`: A list of Python dependencies necessary for running the project.



**Navigate to the utils.py script and adjust the following function calls:**

*Uncomment **main_script_binary()** for binary classification*

*Uncomment **main_script_threeclass()** for advanced multi-class classification*

Run the utils.py script to begin the training 
