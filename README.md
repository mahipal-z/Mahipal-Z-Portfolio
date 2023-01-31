## Data Analysis and Visualization

### [1. Exploratory data analysis of the top 400 video games released since 1977](https://github.com/mahipal-z/Videogames-EDA/blob/main/notebook.ipynb)
----
<ins>__Problem Description:__</ins> The objective was to do a market research analysis of video games sector by identifying release years that users and critics liked best and exploring the business side of gaming by looking at game sales data.  
<ins>__Problem Type:__</ins> __Exploratory Data Analysis (EDA)__  
<ins>__Solution:__</ins> Compared the sales dataset with the critic and user reviews to determine whether video games have improved as the gaming market has grown. Found answers to several business questions using joins, set theory, grouping, filtering, and ordering techniques.  
<ins>__Impact:__</ins> The findings from this analysis provides insights to developer team that helps them finalize the Game Design Document during the pre-production stage.  
<ins>__Tools & Technology:__</ins>  __Relational databases__, __PostgreSQL__  

![](https://github.com/mahipal-z/Mahipal-s-Portfolio/blob/main/images/videogames.jpg)

### [2. Identifying key factors influencing heart disease using 2020 annual CDC survey data of 400k adults](https://www.youtube.com/watch?v=sMiuQQ0Qyz0&ab_channel=MahipalZanakat)
----
<ins>__Problem Description:__</ins> Project’s goal was to build an interactive dashboard from the dataset of patients with heart disease and investigate what factors impact the most in the diagnosis of heart disease.  
<ins>__Problem Type:__</ins> __Dashboard and Reporting__  
<ins>__Solution:__</ins> Developed a live visual report which displays the relationship between likelihood of having a heart disease with factors such as age, gender, lifestyle, etc. using slicers, bar plots, and filters by integrating a Kaggle dataset into Microsoft PowerBI.  
<ins>__Impact:__</ins> The dashboard provides a quick overview of findings from the data set and the accessibility to dig deeper into each section of the analysis.  
<ins>__Tools & Technology:__</ins>  __PowerBI__, Relational Databases  

<img src="https://github.com/mahipal-z/Mahipal-s-Portfolio/blob/main/images/powerbi.PNG" width="550" height="350">

### [3. Retail store KPI Dashboard](https://public.tableau.com/app/profile/mahipal.zanakat/viz/RetailStoreKPIDashboard) 
----
<ins>__Problem Description:__</ins> The goal was to analyse the dataset containing information related to Sales, Profits, and other interesting facts of a superstore giant and publish a dashboard.   
<ins>__Problem Type:__</ins> Dashboard and Reporting  
<ins>__Solution:__</ins> Validated the dataset, integrated with Tableau, analysed data in worksheets and summarised using an overview dashboard.   
<ins>__Impact:__</ins> The report provides a sneak peek of sales and profit numbers in different regions of the United States with filtering ability to categorize and group data.   
<ins>__Tools & Technology:__</ins>  __Tableau__, Donut charts, Maps  

<img src="https://github.com/mahipal-z/Mahipal-s-Portfolio/blob/main/images/tableau.PNG" width="550" height="400">

### [4. Comprehensive analysis of the android app market by comparing over 10,000 apps in Play Store](https://github.com/mahipal-z/playstore-apps-eda/blob/main/Android%20App%20Market%20EDA.ipynb)
----
<ins>__Problem Description:__</ins> The objective was to look for insights in the data to devise strategies to drive growth and retention.  
<ins>__Problem Type:__</ins> Data Analysis and Visualization, __Sentiment Analysis__      
<ins>__Solution:__</ins> I did the exploratory data analysis to answer important business questions on app categories, size, and pricing on Google Play Store. Cleaned and merged tables to explore the user ratings and plotted sentiment polarity score.     
<ins>__Impact:__</ins> The findings from this analysis can help decide what type of apps to develop, how much it should be priced and identify what makes users happy.  
<ins>__Tools & Technology:__</ins>  Relational databases, Python, Pandas, Plotly    

![](https://github.com/mahipal-z/Mahipal-s-Portfolio/blob/main/images/playstore.PNG)

### 5. Simulation of Airport Check-in System  
----
<ins>__Problem Description:__</ins> The goal was to measure the effect of various factors on real time performance of an airport’s check-in operations. Additionally, management wanted to have a strategy that can minimize the operations cost without hurting the customer satisfaction score.    
<ins>__Problem Type:__</ins> Data Visualization, __System Engineering, Operation Research__      
<ins>__Solution:__</ins> Using data collected for a week, I built and executed a model in a simulation software that can report a real-time percentage of passengers satisfied/dissatisfied by using the airport check-in service.
Also, provided an experimenter tool that provides resource allocation solution that improved customer service with minimum cost.      
<ins>__Impact:__</ins> For the given business case, the model __saved $190,000__ in opportunity cost and __$255,000__ of operations cost.  
<ins>__Tools & Technology:__</ins>  Witness Horizon, Witness Action Language


## Machine Learning

### 6. [Blood Donations Prediction](https://github.com/mahipal-z/Blood-donation-project/blob/main/BloodTransfusion.ipynb)
----
<ins>__Problem Description:__</ins> The dataset owners are interested in predicting if past blood donors will donate again during the next collection date as part of an effort to forecast blood supply to mitigate potential supply shortages.      
<ins>__Problem Type:__</ins> __Supervised Learning, Binary Classification__        
<ins>__Solution:__</ins> By performing in-depth data cleaning and training and testing several classifiers, including Logistic Regression and K-Nearest Neighbors with Repeated Stratified K-Fold cross validation, I produced a model that __outperforms__ the ZeroR baseline accuracy by 5.3%.        
<ins>__Impact:__</ins> This model offers the Blood Transfusion Service Center in Hsin-Chu City in Taiwan a more realistic estimate of the number of donors to expect, allowing them to derive a more accurate forecasting of future blood supply and take appropriate action to address potential shortages.    
<ins>__Tools & Technology:__</ins>  Python, Scikit-Learn, Jupyter Notebook  

<img src="https://github.com/mahipal-z/Mahipal-s-Portfolio/blob/main/images/blood.PNG" width="600" height="450">

### 7. [Rating Prediction for a Recipe Website](https://github.com/mahipal-z/Recipe-Rating-Prediction/blob/main/notebook.ipynb)
----
<ins>__Problem Description:__</ins> An online recipe start-up owners wanted to increase website traffic by publishing recipes that are likely to receive high user ratings. To do so, they want to avoid publishing recipes with potentially low ratings. They wanted to know if it is possible to detect recipe ratings using past data.     
<ins>__Problem Type:__</ins> Binary Classification, __Natural Language Processing__        
<ins>__Solution:__</ins> Using NLP techniques to extract features from recipes' text, dimensionality reduction to distil the features, oversampling to balance the dataset, and hyperparameter tuning to identify the best hyperparameters, I developed an XGBoost classification model with an 18% higher recall value than the client's expectations. Based on this model, I identified and provided the client with recommendations to increase website traffic.          
<ins>__Impact:__</ins> The model can help strategize future content publication, forecast user ratings, and increase website traffic.       
<ins>__Tools & Technology:__</ins>  Jupyter Notebooks, DataCamp dataset, Pandas, nltk library, Scikit-learn  

![](https://github.com/mahipal-z/Mahipal-s-Portfolio/blob/main/images/recipe.PNG)  

### 8. [Predicting Credit Card Approvals using a UCI Machine Learning Repository dataset](https://github.com/mahipal-z/Credit-card-approval-prediction/blob/main/notebook.ipynb)
----
<ins>__Problem Description:__</ins> The project's objective was to reduce the time and cost of the credit card application screening process. This cost reduction will allow banking clients to offer online financial advising services for free to new customers opting for an upgraded credit card account.       
<ins>__Problem Type:__</ins> Binary Classification, __Financial Analysis, Risk Analysis__        
<ins>__Solution:__</ins> After one-hot encoding, and mean imputation on dataset, I developed a Logistic Regression model achieving the target classification accuracy using MinMaxScaler and Grid Search cross validation and Found insights that can help provide financial advice to customers for building high credit score.           
<ins>__Impact:__</ins> The model automates the initial screening process of credit card application, saving banks time and resources.         
<ins>__Tools & Technology:__</ins>  Jupyter Notebooks, Pandas, Numpy, Scikit-learn  

![](https://github.com/mahipal-z/Mahipal-s-Portfolio/blob/main/images/credit.PNG)  

### 9. Performance prediction of Laboratory equipment  
----
<ins>__Problem Description:__</ins> Fluid mechanics lab costs much time and capital in running same experiments several times during a semester for educational purpose. We wanted to develop a tool that can predict the performance of centrifugal blower, a lab equipment, accurately to avoid performing actual experiments on it.       
<ins>__Problem Type:__</ins> Supervised Learning, __Linear Regression__          
<ins>__Solution:__</ins> Collected the past experimental data for the equipment and performed some experiments using combinations of different variables. Trained a regression model using a neural network tool and achieved 0.92 of coefficient of determination (R squared).           
<ins>__Impact:__</ins> The tool improved teaching process and learning experience by providing a flexibility to perform experiments in class when Laboratory is unavailable.       
<ins>__Tools & Technology:__</ins>  Visual Gene Developer, __Neural Networks__  
