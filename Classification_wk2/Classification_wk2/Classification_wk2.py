import graphlab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn
except ImportError:
    pass

products = graphlab.SFrame('C:\\Machine_Learning\\Classification_wk2\\amazon_baby_subset.gl\\')
print '# of positive reviews =', len(products[products['sentiment']==1])
print '# of negative reviews =', len(products[products['sentiment']==-1])

import json
with open('C:\\Machine_Learning\\Classification_wk2\\important_words.json','r') as f:
    important_words = json.load(f)
important_words = [str(s) for s in important_words]
print important_words

def remove_punctuation(text):
    return text.translate(None,string.punctuation)

products['review_clean'] = products['review'].apply(remove_punctuation)
for word in important_words:
    products[word] = products['review_clean'].apply(lambda s:s.split().count(word))
products['perfect']
products['contains_perfect'] = products['perfect'].apply(lambda x: 1 if x >= 1 else 0)
print products['contains_perfect'].sum()

def get_numpy_data(data_sframe, features, label):
    data_sframe['intercept'] = 1
    features = ['intercept'] + features
    features_sframe = data_sframe[features]
    feature_matrix = features_sframe.to_numpy()
    label_sarray = data_sframe[label]
    label_array = label_sarray.to_numpy()
    return(feature_matrix, label_array)

# Warning: This may take a few minutes...
feature_matrix, sentiment = get_numpy_data(products, important_words, 'sentiment') 
feature_matrix.shape
len(important_words)

'''
produces probablistic estimate for P(y_i = +1 | x_i, w).
estimate ranges between 0 and 1.
'''
def predict_probability(feature_matrix, coefficients):
    # Take dot product of feature_matrix and coefficients  
    # YOUR CODE HERE
    wh = np.dot(feature_matrix,coefficients)
    # Compute P(y_i = +1 | x_i, w) using the link function
    # YOUR CODE HERE
    predictions = 1 / ( 1 + np.exp(-wh))
    #print predictions
    # return predictions
    return predictions

dummy_feature_matrix = np.array([[1.,2.,3.], [1.,-1.,-1]])
dummy_coefficients = np.array([1., 3., -1.])

correct_scores      = np.array( [ 1.*1. + 2.*3. + 3.*(-1.),          1.*1. + (-1.)*3. + (-1.)*(-1.) ] )
correct_predictions = np.array( [ 1./(1+np.exp(-correct_scores[0])), 1./(1+np.exp(-correct_scores[1])) ] )

print 'The following outputs must match '
print '------------------------------------------------'
print 'correct_predictions           =', correct_predictions
print 'output of predict_probability =', predict_probability(dummy_feature_matrix,dummy_coefficients)

def feature_derivative(errors, feature):     
    # Compute the dot product of errors and feature
    derivative = ...
    
    # Return the derivative
    return derivative