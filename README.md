# Meli Data Challenge 2021

https://ml-challenge.mercadolibre.com/

## Goal of the project

The task is to predict how long it will take for the inventory of a certain item to be sold completely. 

In the evaluation id given the item target stock, and t will provided a prediction for the number of days it will take an item to run out. Possible values range from 1 to 30. Rather than giving a point estimate, it is expected to provide a score for each the possible outcomes.


## Exploratory data analysis

Checkout the preprocessing Notebook. 


### Strategy 

Data was pre-processed with Pyspark. XGboost with a softmax objective was used as model.
Hyperparameters were optimized with Ray Tune. 


## Execution

- Download the input files from 
https://ml-challenge.mercadolibre.com/rules, and place them in the DATA folder

* Install the dependencies with pip via the requirements.txt file located in the main directory

>  pip install -r requirements.txt

* Then run the main.py:

> python main.py 

## Results
A csv with the outcome of the analysis will be generated in the DATA folder.