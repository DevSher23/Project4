# The Pink Avengers
![image](https://user-images.githubusercontent.com/108558769/221731108-684a4603-eeab-4d7f-adcc-9bdfaea6d5ff.png)
# Objective:
### Every year during October, Breast Cancer Awareness campaigns are put together to increase the awareness of the disease. Many forget to take the necessary steps to have a plan to detect the disease in its early stages. We plan to use the information gathered in this study to automatically detect if a breast tumor is malignant or benign. We will use machine learning to help with early diagnosis of breast cancer by analyzing the characteristics found in the digitized images of hundreds of tumors to determine whether they are malignant or benign. 

<img width="644" alt="Screen Shot 2023-02-27 at 8 57 57 PM" src="https://user-images.githubusercontent.com/108558769/221733221-d5fb3cd9-5994-4e21-b2ca-a99a78515cd4.png">

# Outline
### We utilized Jupyter Notebook and Pandas to prepare and clean our data.

<img width="1161" alt="Screen Shot 2023-03-02 at 1 27 12 PM" src="https://user-images.githubusercontent.com/108558769/222519338-c67c3681-fcd8-443b-9fdd-80347b48454c.png">

<img width="498" alt="Screen Shot 2023-03-02 at 1 27 41 PM" src="https://user-images.githubusercontent.com/108558769/222519474-95b2e376-2586-42e0-a00e-8359dfd4f2d0.png">

* As shown above, columns 1-30 contain 30 real-value features that have been computed from digitized images of the cell nuclei, which can be used to build a model to predict whether a tumor is benign or malignant. Columns 1-10 indicate Mean values, 11-20 indicate Standard Error (se) and 21-30 indicate Worst values.
* Attribute Information: - radius (mean of distances from center to points on the perimeter) - texture (standard deviation of gray-scale values) - perimeter - area - smoothness (local variation in radius lengths) - compactness (perimeter^2 / area - 1.0) - concavity (severity of concave portions of the contour) - concave points (number of concave portions of the contour) - symmetry - fractal dimension ("coastline approximation" - 1)
* The mean, standard error, and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.


<img width="441" alt="Screen Shot 2023-03-02 at 1 28 31 PM" src="https://user-images.githubusercontent.com/108558769/222519516-12d2e4cc-9748-4feb-9939-2b25e76c0b5d.png">

<img width="880" alt="Screen Shot 2023-03-02 at 1 33 43 PM" src="https://user-images.githubusercontent.com/108558769/222520312-20664d5f-5eee-41ea-b639-78acbbdebb63.png">
 
### Postgres was used to load our database. 

<img width="884" alt="Screen Shot 2023-03-02 at 1 31 56 PM" src="https://user-images.githubusercontent.com/108558769/222520004-89a14337-680c-459e-a19f-4f3d59edf5a1.png">

### For our visualizations, we utilized Tableau and matplotlib.

https://public.tableau.com/app/profile/anna1103/viz/Cancer_Prediction_ML/Story1?publish=yes

<img width="673" alt="Screen Shot 2023-03-06 at 8 09 28 PM" src="https://user-images.githubusercontent.com/108558769/223293103-23a7282d-61f5-4235-a3e7-87913d7b9b00.png">


### Correlation Coefficient
<img width="789" alt="Screen Shot 2023-03-02 at 2 44 24 PM" src="https://user-images.githubusercontent.com/108558769/222610875-621a1ac6-1b12-42d7-a813-300040c80a7a.png">

### Correlation Matrix
<img width="701" alt="Screen Shot 2023-03-02 at 2 44 41 PM" src="https://user-images.githubusercontent.com/108558769/222610925-5e07f580-f050-4c5b-9610-5f2120fcfdba.png">

### Scatter Plot

### Texture Mean vs Radius Mean

<img width="564" alt="Screen Shot 2023-03-06 at 7 09 40 PM" src="https://user-images.githubusercontent.com/108558769/223285088-35d471c2-2bb1-4079-93b9-59a824a496a3.png">
    
    *Radius mean/texture mean- Both of these variables could help to visualize 
    any patterns or trends in the data and identify any clear separation between 
    the two diagnosis categories and to see if there is a clear pattern or trend 
    that distinguishes benign from malignant samples. 

### Concavity Mean vs Area Mean

<img width="590" alt="Screen Shot 2023-03-06 at 7 10 23 PM" src="https://user-images.githubusercontent.com/108558769/223285165-07a33159-cf81-443d-9a1d-9db90c805ad0.png">

    *Area mean/concavity mean - "area_mean": This variable represents the 
    mean area of the breast mass sample. It is also measured on a continuous 
    scale and could be a relevant variable for breast cancer diagnosis, as larger 
    masses may be more likely to be malignant."concavity_mean": This variable 
    represents the mean severity of concave portions of the contour of the 
    breast mass sample. It is also measured on a continuous scale and could be 
    a relevant variable for breast cancer diagnosis, as more severe concavity 
    may be more indicative of malignancy.

# Data Model Machine Learning & Data Optimization Machine Learning

### Python script TO evaluate two models:

* Random Forest optimizing with RandomizedSearchCV
    
<img width="340" alt="Screen Shot 2023-03-06 at 9 44 50 PM" src="https://user-images.githubusercontent.com/108558769/223306886-5d811ef6-0d16-4f6b-99b7-f404f6c5fbe5.png">

* Logistic Regression optimizing with GridSearchCv

<img width="343" alt="Screen Shot 2023-03-06 at 9 45 31 PM" src="https://user-images.githubusercontent.com/108558769/223307004-dcf45326-8caa-4c84-a7cf-f1df2a01c360.png">

* Our optimization was aimed to reduce false negatives. Below is the confusion matrix for the logistic regression
    
<img width="264" alt="Screen Shot 2023-03-06 at 9 47 41 PM" src="https://user-images.githubusercontent.com/108558769/223307304-b293a342-fc46-4eb5-912c-0a146fbcdf63.png">

<img width="296" alt="Screen Shot 2023-03-06 at 9 48 15 PM" src="https://user-images.githubusercontent.com/108558769/223307396-349ba3a4-6d7b-4c7d-93c2-288fc8c46d0c.png">

# Our Research Questions:

        *1. Could we detect if a breast tumor is malignant or benign?
            - Yes, we can detect if a breast tumor is malignant or benign. Overall, the
            variables we used in our breast cancer testing is very helpful in the aid of
            identifying potential biomarkers/predictors for breast cancer. 
            
        *2. Which characteristics are most relevant? 
            - size
            - shape
            - texture
            
        *3. Which were the best parameters used during optimization?
            - Radius_worst
            - Fractal_dimension_mean
            - Concave point_worst
            - Concavity_mean 
            - Compactness_worst


# AS The Pink Avengers...We hope our efforts of using ML models to detect Breast Cancer, helps save the lives of breast cancer patients!
![image](https://user-images.githubusercontent.com/108558769/221735330-c802a8c1-2527-4f2e-b189-857fa361fe9b.png)


### Our Dataset:
        *https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
        
        

