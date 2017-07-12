Files submitted .

1. Readme
2. Project Report
3. ml.scala - scala code
4. ML.r -  R code
5. output_log_scala.txt - log output of the scala code run on Databricks
6. Data Folder containing the test and train files and the submission(submit.csv) made to the competition

HOW TO EXECUTE

1. Scala

Instructions for executing the scala code in Databricks-
- Create a Spark Cluster version - Spark 2.0.2 (scala 2.11)
- create a Scala notebook and attach the code to the cluster
- Load the test and train files in Databricks table. Replace the path of the train and test .csv in code with the new path thats created
- Click Run-all to run the code.


2.R

R version used:
R 3.3.2

PACKAGES USED:
caret 

lattice 

pROC 


ggplot2
 

party
 
gbm 


elmNN

COMMAND FOR COMMAND LINE EXECUTION(windows):
Rscript --vanilla ML.R train.csv test.csv

STEPS TO EXECUTE CODE:
-> Update the location of the train.csv and test.csv files in the ML.R program to your local path.
-> Package required: caret
, lattice, 
pROC, 

ggplot2, 

party, 
gbm
, 
elmNN
-> The program folder contains the files, train.csv test.csv.
-> Traverse to the location containing the submission files and execute the above mentioned file

ASSUMPTIONS:
-> The last attribute will be the classifying attribute of our dataset.

-> Number and sequence of attributes needs to remain the same. We have attached test.csv and train.csv along with our program for execution purposes.
