# Data Cleaning & Analysis of FIFA Salaries
***
kaggle often provides clean, easy to use datasets for model training and visualization. The
purpose of this project is to take a messier dataset, clean it, explore the data, and perform 
basic statistical analyses. The main aim is not to produce the best possible model, but rather 
this project focuses on the steps leading up to model training. The dataset can be retrieved from:

[FIFA 21 messy, raw dataset for cleaning/exploring](https://www.kaggle.com/datasets/yagunnersya/fifa-21-messy-raw-dataset-for-cleaning-exploring)

I also used dagshub for data versioning and storage of processed data and metrics:

[dagshub](https://dagshub.com/philjhowson/Data_Cleaning_FIFA)

## Project Organization
------------------------------------------------------------------------
    root
    ├── .dvc #contains the dvc config file 
    ├── data
    │    ├── processed_data #dvc tracked folder with processed files
    │    └── raw_data #stores the raw data file
    ├── images #stores output images from python scripts
    ├── metrics #dvc tracked metrics for models
    ├── models #dvc tracked model saves
    ├── src #python code
    ├── streamlit #contains files and assets associated with streamlit
    ├── .dvcignore
    ├── .gitignore
    ├── README.md
    ├── dvc.lock #dvc data versioning
    ├── dvc.yaml #dvc pipeline for reproducability
    └── requirements.txt