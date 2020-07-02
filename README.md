# Talpa coding_challenge
## Generate predictions for Mining data

_Repository for generating predictions for Mining data_.

### Installation instruction :

Clone the repository and install talpa using:

"python setup.py install"  Or  "python setup.py sdist"


#### General Instructions :
1. Datasets must be stored in folder "dataset"
2. Inside `minning_data.py` file as shown below
    - Provide the dataset filename 
    - Provide the classifier name
3. Execute `minning_data.py` file
```
if __name__ =='__main__':                                     
    start = time.time()                                       
    filename = 'data_case_study.csv'    ## Provide the dataset filename                         
    model_name = "LogisticRegression"   ## Provide the classifier name                      
                                                              
    read_data = DatasetReader("dataset", filename)            
    read_data.check_data_validity(model_name)                 
    print("Execution time taken:", time.time()-start, "sec")              
```

Execution time taken by each classifier:

|RandomForest|LogisticRegression|GradientBoost|XGBoost|KNN|
|---|---|---|---|---|
|22.7308 sec|51.6076 sec|631.827 sec|188.0264 sec|688.119 sec|

**Authors / Maintained by: Priyanka Roy**
