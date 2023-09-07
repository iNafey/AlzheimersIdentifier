# Alzheimer's identification using CNN

## My Work
I'm using a machine learning framework called 'keras' provided by tensorflow to develop my CNN model to detect AD. I also use transfer learning models for performance comparison, Grad-CAM heatmap and a web based UI for real users to interact with the custom model.

## Git Contents
Currently, all files are in the root directory.

* My gantt charts (within the ```Gantt Charts``` directory) and time plans can be found in various formats: pdf, png, and gan.
* Within the FinalYrProject folder there are 2 versions of the final project. This is because the user can either choose to run the notebook on Kaggle (highly recommended) or on their local machine. It's preferred ```kaggle_ad_classifier.ipynb``` is used by importing it via Kaggle (instructions provided below). ```ad_classifier.ipynb```  file is there to be ran on a local machine. The main reason for the two notebooks is for file handling reasons (different OS and directory structures). Both of the notebooks contain the custom CNN model and other models for comparison and performance optimisation.
* There's also a UI using flask (backend) and html/css/js (frontend) to interact with the AD classifier which is powered by the custom CNN. ```predict_app.py``` and ```static``` folder store this software.
* There are 2 datasets, ```AlzheimersDataset``` is an imbalanced dataset used for the first part of the project. ```Combined Dataset``` is the main dataset which has been balanced using deep learning augmentation techniques. Both datasets include brain MRI scans provided by Kaggle.
* ```logs``` folder stores the output when the model is training allowing it to store weights, pause, resume training. It's used by a callback function to then evaluate performance.
* ```models``` are the CNN models built during the project. They are stored so they can be loaded and used for ensemble learning approaches.

## Installation instructions

Since, the software system has two parts: CNN models built in python and a web based UI, the instructions will be divided into 2 sections.

## Section 1: Main software system (to be ran as a python notebook)

### Kaggle Installation Instructions (highly recommended):
It is highly recommended you run the notebook on Kaggle as it has great hardware resources like 13GB RAM and P100 Nvidia GPUs.

	1. Register a Kaggle account (via https://www.kaggle.com/ ) using a valid email.
	2. Once you verify your email, head over to the setting (click on profile in the top right of the page) and complete phone verification. (It' an authentication step to prevent scam and fraud)
	3. Click on 'Create' --> 'New Notebook'  (on the left sidebar)
	4. Then perform the steps: File > Import Notebook > Upload the python notebook 'kaggle_ad_classifier' found in the FinalYrProject directory on this git repo.
	5. There are 2 datasets used in this project (both provided by Kaggle). Head over to the 'Add Data' button and search for 'Alzheimer's Dataset ( 4 class of  Images)' and add it. Similarly, also search for 'Best Alzheimer MRI dataset (99% accuracy)' to add the improved and balanced version of the previous dataset.  
	6. To start the notebook, look for the 3 vertical dots (next to power icon) and change the 'Accelerator' from None to GPU P100. Then, in the notebook options section on the right of the page, turn on the internet. Then, click on the Power Icon to switch on the kernel.
	7. Finally, run the cells manually (in order) to view the project. The notebook structure is divided into two sections: Dataset 1 (Imbalanced) and Dataset 2 (Balanced).   
	8. IMPORTANT NOTE: Make sure to carefully run the cells as there are manual instructions for file handling (due to different directory structures between local machine and Kaggle) if you used a Kaggle dataset or ran it on a local machine.

### Local Machine Installation Instructions:

#### **Hardware Requirements:**
	- 16GB RAM minimum
	- 5GB free storage
	- Some Nvidia GPU (much preferred else training deep learning models will take hours)

#### **Software Requirements:**

	- Jupyter Notebook (Can be with an Anaconda environment or standalone)
	- Python 3.7.12 minimum (Also tested with 3.9.13 on my local machine)
	- Libraries mainly for machine learning and visualization as defined below (They take up about 2GB of storage space, mainly Tensorflow)

#### **Library installs (required):**

IMP: the libraries mentioned in the imports must be installed in the root directory (current folder) using ```pip install $library_name```.

You must have pip to install all the python packages, so first install pip (in case you don't, if you have python installed then pip should be too) using:

``` python -m pip install -U pip ```

See the documentation if there are issues with pip installation: https://pip.pypa.io/en/stable/installation/

If you don’t have numpy already (if you have python then you most likely will do) then do:
``` pip install numpy ```


To install Tensorflow do:

```pip install tensorflow==2.11.*``` (my local machine uses 2.11.0 and Kaggle notebooks use 2.6.0)
OR
```pip install tensorflow==2.12.*``` (not tested)

To install keras-tuner do:
```pip install keras-tuner```

To install imbalanced-learn run:
```pip install imbalanced-learn```

To install flask and flask cors do:
```pip install -U flask flask-cors```

To install sklearn run:
```pip install -U scikit-learn```

To install matplot-lib run:
```python -m pip install -U matplotlib```


## Section 2: AD predictor in a web based app

### IMPORTANT: Make sure Flask framework is installed using ```pip install -U flask flask-cors``` if you have not already done so.

1. In the ```FinalYrProject``` directory, run ```export FLASK_APP=predict_app.py```
2. Then run the command ```flask run``` which initialises the web app on localhost
3. Head over to a browser (preferably Google Chrome) and go to ```http://localhost:5000/static/index.html```.
4. Upload a MRI brain scan in JPEG format ONLY. Retrieve an image from the ```AlzheimersDataset/test``` folder by navigating through the file upload within the app. Click predict to view results and stats.
