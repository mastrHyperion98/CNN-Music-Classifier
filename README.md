# MachineLearningProject

Supervised machine learning program that will efficiently classify new unseen music tracks by genre. The project will train a Neural Network model on a database of 4,000 categorized music files. It will take the music files as input and extract relevant features such that we can accurately predict the genre of new tracks that are inputted into the syste

## Installation

### Python Virtual Environment (Recommended) 

#### Create an environment

Create _venv_ folder within the project folder, if not included.

For Linux and Mac:

    $ cd MachineLearningProject
    $ python3 -m venv venv

On Windows:

    > py -3 -m venv venv

#### Activate the environment

Before you work on the project, activate the corresponding environment.

For Linux and Mac:

    $ source venv/bin/activate

On Windows:

    > venv\Scripts\activate

#### Install the dependencies


    pip3 install -r requirements.txt
    
    
    
##### Submission Edits: 

The above description explains how to install the dependencies and setup a venv environment that we use for the development of the project. 
However, depending on your development environment the process to get the project opened and working will differ. 

Recommended IDE: Pycharm Community/Professional.
 
It is strongly recommend that pycharm be used for running the project. It should be as simple as: 

* Opening the directory with Pycharm
* Create a venv interpreter via File/Settings -> Project:MachineLearningProject -> Python interpreter
* Either use the above commands to activate the venv environment and run the command to install the dependencies while being CD into the venv directory.
* Sometimes Pycharm will recognize the requirement.txt file and prompt you to install all the missing dependecies. Accept and wait for Python to be done. 


###### Execution Order: 
* The project will be submitted with all the required files to simply execute main.py. However. NO TRACKS WILL BE PROVIDED because of data sizes constraints and the time required to extract data from them.
All the paths (dataset) folder will be included. You can manually create a folder and insert an audio file in a ogg format if you want to test the DataExtraction.main.py BUT it most be at least 90seconds in length
* main.py will execute three different experiments using our smallest datasets. The smallest dataset are the ones that execute most quickly and require less system resources. Even our 
800 spectrogram images sample takes about 30min to train on an 8 core Ryzen 7 processor and requires more than 14 GB of system memory. We are working with a lot of data. Our 
4k datasets are omitted as the version required to run on the CNN takes several hours to run and over 40GB of system memory. 
* data_spectrogram_160_10sec.csv is the default file for the CNN and data_800_10sec.csv is the default file for the FFNN and random forest. These can be changed but it is not recommened. 
* Running main.py as is without any modifications will create a large amount of graphs and csv output files contain various analytical outputs. Those can be compared with what we have in 
our report. However, please keep in mind that the best data presented in the report were generated using the 4000 track dataset. Those were trained overnight!!!! 
* I will also rename the CNN_Test_Analytics.csv file that experiment three generates to add SAMPLE at the beginning. Random_sample = 0 is enabled and as such your run should match our data
in the sample. 

IT IS HIGHLY DISCOURAGED TO TRY AND RUN BIGGER DATASETS. NOT MEETING THE HARDWARE REQUIREMENTS WILL CAUSE SYSTEMS TO CRASH, HANG AND BECOME UNRESPONSIVE! 


##### LIBRARY REQUIREMENTS

* numpy==1.18.0
* pandas==1.1.4
* SciPy==1.4.1
* joblib
* threadpoolctl==2.0.0
* scikit-learn==0.23.2
* librosa==0.8.0
* matplotlib==3.3.2
* torch==1.7.0
* pytorch_lightning

GPU is not required to run the provided data sets
RECOMMEND: 16 GB of Memory MINIMUM --> Will need to comment out training analytics ouput for Experiment 3 when using spectrogram_800_15sec.csv as data for the CNN_Test_Analytics

Depending on the dataset and system hardware training can take anywhere from 10min to an hour. 
Jupyter Notebook not used as it is a nightmare for cooperative/simultaneous work


spectrogram_800_15sec google drive link: https://drive.google.com/file/d/1FFShzGx2bJitrsTCp_5tECbKlsgqJnIN/view?usp=sharing 
warning ~1.5Gb download
