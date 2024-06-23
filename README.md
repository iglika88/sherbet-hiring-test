# NLP technical test for the SHERBET project

## Goal

Automatically classify newspapers according to their category by using a computational approach.

## Dataset

The corpus to be classified is available in the file `dataset.csv`.
The file contains a set of documents extracted from English newspapers. Each document is labeled by an ID, and has a content and a category (field `type`).
There are 5 different categories available [`business`, `entertainment`, `politics`, `sport`, `tech`] that should be predicted. 

## Steps to accomplish the task
The goal of the text is to **build a machine learning model to automatically predict the field type (i.e. the category)**.


1. Load the data from the csv file located in `dataset.csv`.
2. Embed the text within a numerical space using the method of your choice.
3. Present a 2D or 3D graph accurately reprensting the dataset.
4. Train a classifier on a subset of the dataset (train subset).
5. Evaluate this classifier on the held-out subset (test subset) and return the most interesting metric for quality evaluation of this multi-class model.


