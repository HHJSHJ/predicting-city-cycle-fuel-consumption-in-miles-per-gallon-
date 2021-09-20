# predicting city-cycle fuel consumption in miles per gallon ( in terms of 3 multivalued discrete and 5 continuous attributes.)
'''
We are using the Auto MPG dataset from the UCI Machine Learning Repository. Here is the link to the dataset:

http://archive.ics.uci.edu/ml/datasets/Auto+MPG
The data concerns city-cycle fuel consumption in miles per gallon, to be predicted in terms of 3 multivalued discrete and 5 continuous attributes.
'''
Problem Statement — The data contains the MPG (Mile Per Gallon) variable which is continuous data and tells us about the efficiency of fuel consumption of a vehicle in the 70s and 80s.

Our aim here is to predict the MPG value for a vehicle, given that we have other attributes of that vehicle.
Exploratory Data Analysis with Pandas and NumPy:
          For this rather simple dataset, the exploration is broken down into a series of steps:

          Check for data type of columns
          ##checking the data info
          data.info()
          Check for null values.
          ##checking for all the null values
          data.isnull().sum()
          The horsepower column has 6 missing values. We’ll have to study the column a bit more.

          Check for outliers in horsepower column
          ##summary statistics of quantitative variables
          data.describe()

          ##looking at horsepower box plot
          sns.boxplot(x=data['Horsepower'])
          The horsepower column has 6 missing values. We’ll have to study the column a bit more.

          Check for outliers in horsepower column
          ##summary statistics of quantitative variables
          data.describe()

          ##looking at horsepower box plot
          sns.boxplot(x=data['Horsepower'])
          etc...
There are many ways to split the data into training and testing sets but we want our test set to represent the overall population and not just a few specific categories. Thus, instead of using simple and common train_test_split() method from sklearn, we use stratified sampling.

Stratified Sampling — We create homogeneous subgroups called strata from the overall population and sample the right number of instances to each stratum to ensure that the test set is representative of the overall population.
Data Preparation using Sklearn
One of the most important aspects of Data Preparation is that we have to keep automating our steps in the form of functions and classes. This makes it easier for us to integrate the methods and pipelines into the main product.

Here are the major tasks to prepare the data and encapsulate functionalities:
         Preprocessing Categorical Attribute — Converting the Oval
         Data Cleaning — Imputer
         We’ll be using the SimpleImputer class from the impute module of the Sklearn library
         Attribute Addition — Adding custom transformation
         In order to make changes to datasets and create new variables, sklearn offers the BaseEstimator class. Using it, we can develop new features by defining our own class.

         We have created a class to add two new features as found in the EDA step above:
         acc_on_power — Acceleration divided by Horsepower
         acc_on_cyl — Acceleration divided by the number of Cylinders
         Setting up Data Transformation Pipeline for numerical and categorical attributes
         As I said, we want to automate as much as possible. Sklearn offers a great number of classes and methods to develop such automated pipelines of data transformations.

        The major transformations are to be performed on numerical columns, so let’s create the numerical pipeline using the Pipeline class.
        In the code , we have cascaded a set of transformations:
        Imputing Missing Values — using the SimpleImputer class discussed above.
        Custom Attribute Addition— using the custom attribute class defined above.
        Standard Scaling of each Attribute — always a good practice to scale the values before feeding them to the ML model, using the standardScaler class.
        Combined Pipeline for both Numerical and Categorical columns
        We have the numerical transformation ready. The only categorical column we have is Origin for which we need to one-hot encode the values.
        Here’s how we can use the ColumnTransformer class to capture both of these tasks in one go.
        To the instance, provide the numerical pipeline object created from the function defined above. Then call the OneHotEncoder() class to process the Origin column.

Final Automation
        With these classes and functions defined, we now have to integrate them into a single flow which is going to be simply two function calls.
        Preprocessing the Origin Column to convert integers to Country names.
Selecting and Training Machine Learning Models
        Since this is a regression problem, I chose to train the following models:
        Linear Regression
        Decision Tree Regressor
        Random Forest Regressor
        SVM Regressor
  It’s a simple 4-step process:
      Create an instance of the model class.
      Train the model using the fit() method.
      Make predictions by first passing the data through pipeline transformer.
      Evaluating the model using Root Mean Squared Error (typical performance metric for regression problems)
      
Cross-Validation and Hyperparameter Tuning using Sklearn
       Now, if you perform the same for Decision Tree, you’ll see that you have achieved a 0.0 RMSE value which is not possible – there is no “perfect” Machine Learning Model (we’ve not reached that point yet).
       Problem: we are testing our model on the same data we trained on, which is a problem. Now, we can’t use the test data yet until we finalize our best model that is ready to go into production.
       Solution: Cross-Validation
       Scikit-Learn’s K-fold cross-validation feature randomly splits the training set into K distinct subsets called folds. Then it trains and evaluates the model K times, picking a different fold for evaluation every time and training on the other K-1 folds.
       The result is an array containing the K evaluation scores. Here’s did for 10 folds.
Fine-Tuning Hyperparameters
        After testing all the models, you’ll find that RandomForestRegressor has performed the best but it still needs to be fine-tuned.
        A model is like a radio station with a lot of knobs to handle and tune. Now, you can either tune all these knobs manually or provide a range of values/combinations that you want to test.
        We use GridSearchCV to find out the best combination of hyperparameters for the RandomForest model.
 At the end,checking feature performance and evaluation .
