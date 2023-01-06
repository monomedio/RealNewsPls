# News Headline Classifier
Decision tree classifier for predicting "real" and "fake news" headlines. 
Implemented as part of an assignment in CSC311, a course offered at the University of Toronto.

The fake news headlines are from [here](https://www.kaggle.com/mrisdal/fake-news/data), while the real news headlines are from [here](https://www.kaggle.com/therohk/million-headlines.)

The data were cleaned by removing words from fake news titles that are not a part of the headline, removing special characters from the headlines, and restricting real news headlines to those after October 2016 containing the word “trump”.
The script used for this cleaning is `clean_script.py` and was included as part of the assignment.
