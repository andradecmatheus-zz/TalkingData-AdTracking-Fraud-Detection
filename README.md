<h1 align="center">
TalkingData AdTracking FraudDetection
</h1>

<h4 align="center">
Building a machine learning model to determine whether a click is fraudulent or not
</h4>




### :bookmark_tabs: Description

Fraud risk is everywhere, but for companies that advertise online, click fraud can happen at an overwhelming volume, resulting in misleading click data and wasted money. Ad channels can drive up costs by simply clicking on the ad at a large scale. With over 1 billion smart mobile devices in active use every month, China is the largest
mobile market in the world and therefore suffers from huge volumes of fradulent traffic.

TalkingData, China’s largest independent big data service platform, covers over 70% of active mobile devices nationwide. They handle 3 billion clicks per day, of which 90% are potentially fraudulent. Their current approach to prevent click fraud for app developers is to measure the journey of a user’s click across their portfolio, and flag IP addresses who produce lots of clicks, but never end up installing apps. With this information, they've built an IP blacklist and device blacklist.

While successful, they want to always be one step ahead of fraudsters and have turned to the Kaggle community for help in further developing their solution. In their 2nd competition with Kaggle, you’re challenged to build an algorithm that predicts whether a user will download an app after clicking a mobile app ad.



### :dart: Goal
In summary, the goal is to build a machine learning model to determine whether a click for mobile app ads is fraudulent or not.

It is used R Language for the construction of this project.



### :exclamation: ​Instructions

1. Datasets: 'train_sample.csv' contains 100k rows, it is the input data for building the model. And 'test.csv' is found at problem's page on Kaggle.

2. EDA: it contains the 'ExploratoryDataAnalysis' on train_sample data, in which we investigate the dataset in order to know it, generating plots and correlations.

3. FeatureEngineering:  it performs the data munging function, that's necessary for build the model and for submit the solution on kaggle.

4. Modelling_and_Evaluation: after evoke 'FeatureEngineering' function:
   - it generates the train and test datasets;

   - it applies random under and over sampling;

   - it builds the undersampled and oversampled models, and evaluete them.

   - it optimizes the best model.

   - it generates the solution to send on Kaggle.

     

### Useful links

- [Problem's page on Kaggle](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/overview) 

- [Big Data Analytics with R and Microsoft Azure Machine Learning](https://www.datascienceacademy.com.br/course/analise-de-dados-com-r) (this repository is a project for data science course from Data Science Academy)

