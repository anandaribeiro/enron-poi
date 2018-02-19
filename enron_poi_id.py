#!/usr/bin/python

import sys
import pickle
#from tester import dump_classifier_and_data
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

sys.path.append("tools_by_Udacity")
from feature_format import featureFormat, targetFeatureSplit

###################################################
#Functions
###################################################

#Function to create a new feature for the dataset:
def new_feature(data):
    extras = float(data['bonus']) + float(data['long_term_incentive']) +\
            float(data['other']) + float(data['expenses'])
    extras_plus_salary = extras + float(data['salary'])
    #If the denominator is zero or NaN, the new feature will receive the 
    #value zero.
    if (np.isnan(extras_plus_salary)) | (extras_plus_salary <= 0):
        data['extras_over_salary'] = 0
    else:
        data['extras_over_salary'] = extras / (extras_plus_salary)
    return data['extras_over_salary']

#Function to build histograms splitting the data into POIs and non-POIs
def build_histogram(data, variable):   
    pois = data[data['poi'] == 1][variable]
    non_pois = data[data['poi'] == 0][variable]    

    plt.hist([non_pois, pois], bins = 20, label = ['Non-POI','POI'])
    plt.xlabel(variable)
    plt.ylabel("Count")
    plt.legend()
    plt.show()
    plt.close()

#This function contains the script used for the data exploration.   
def explore_data():
    #Number of indivuduals in the dataset:
    print "Total number of individuals in the original dataset:", len(data_dict.keys())
    
    #Counting NaN values in each feature.
    #The following for-loop was also used to organize the data in a dataframe,
    #in order to build some plots.
    
    features_missing_values = {}
    features_data = defaultdict(list)
    for name in data_dict.keys():     
        for feature in data_dict[name].keys():
            value = data_dict[name][feature]
            if value == 'NaN':
                if feature in features_missing_values.keys():
                    features_missing_values[feature] += 1
                else:
                    features_missing_values[feature] = 1
            #All variables are transformed to float, except for POI and e-mail
            if (feature != 'poi') and (feature != 'email_address'):
                value = float(value)
            #E-mail will not be recorded in this dataframe since it is not 
            #interesting for this data exploration
            if feature != 'email_address':
                features_data[feature].append(value)
    
    print ""
    print "Number of missing values in each feature:\n"
    print features_missing_values
    
    features_data = pd.DataFrame(features_data)
    #All lines with all variables equal to NaN will be removed and the remaining NaN
    #will be transformed to zero.
    #If there is at least 2 non-NaN values (poi and one more), the row will 
    #not be dropped.
    features_data.dropna(thresh = 2, inplace = True)
    features_data.fillna(0, inplace = True)
    print ""
    print 'Number of individuals after removing lines with all values NaN:', \
        len(features_data.index)
    print ""
    print 'Number of POIs:', len(features_data[features_data['poi'] == 1].index)
    print 'Number of non POIs:', len(features_data[features_data['poi'] == 0].index)
     
    #A new feature will be created: extras over salary
    #Building the new feature:
    features_data['extras_over_salary'] = features_data.apply(new_feature, axis = 1)
    
    plt.ioff()
    #A heatmap of the correlation between variables will be made.
    #The features with too many missing values will be not used in the following plots.
    #Features to remove from the plot:
    features_to_remove = ['deferral_payments','deferred_income', 'director_fees',
                          'loan_advances', 'restricted_stock_deferred']
    corr = features_data.drop(features_to_remove, axis = 1).corr() 
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Heatmap of Correlation between Features')
    plt.show()
    
    #A scatterplot matrix will be build but not shown. 
    #It will be saved in a figure.  
    scatterplot_matrix = sns.pairplot(features_data.drop(features_to_remove, axis = 1),
                                      hue="poi", diag_kind="kde")
    scatterplot_matrix.savefig("scatterplot_matrix.png", bbox_inches='tight')
    plt.close('all')
    
    #The new feature will be further explored with an histogram:
    build_histogram(features_data, 'extras_over_salary')
    
    #A Histogram will also be made for from_messages.
    build_histogram(features_data, 'from_messages')
    print 'Investigation of some cases that seem outliers \n'
    print'Case 1: Bonus above $ 6.000.000:\n'
    print features_data[features_data['bonus'] > 6000000]
    print ""
    print "Case 2: Expenses above $ 200.000 \n"
    print features_data[features_data['expenses'] > 200000]
    print ""
    print "Case 3: From messages above 12500\n "
    print features_data[features_data['from_messages'] > 12500] 
    print ""
    print "Case 4: from_poi_to_this_person above 500\n "
    print features_data[features_data['from_poi_to_this_person'] > 500] 
    print ""
    print "Case 5: Long term tncentive above $ 5.000.000 \n "
    print features_data[features_data['long_term_incentive'] > 5000000] 
    print ""
    print "Case 6: Negative restricted stock \n "
    print features_data[features_data['restricted_stock'] < 0] 
    print ""
    print "Case 7: Other above $ 6.000.000 \n "
    print features_data[features_data['other'] > 6000000] 
    print ""
    print "Case 8: To messages above 10000\n "
    print features_data[features_data['to_messages'] > 10000] 
    print ""
    print "Case 9: total_payments above $ 100.000.000 \n "
    print features_data[features_data['total_payments'] > 100000000] 
    print ""
    print "Case 10: total_stock_value above $ 40.000.000 \n "
    print features_data[features_data['total_stock_value'] > 40000000] 
    print ""
    
    #The variable names was used to compare the index of each result with
    # the names of the individuals in the dataset
    names = data_dict.keys()

#Function to apply cross validation with GridSearchCV:
def kfold_cross_validation(pipe, param_grid, features, labels):
    #This function does a K-fold cross-validation with 6 folds
    cv = StratifiedShuffleSplit(n_splits = 20, random_state = 0, 
                                train_size = 0.70)
    gs = GridSearchCV(pipe, param_grid=param_grid, 
                       scoring = ['recall', 'precision', 'accuracy'],
                       refit = 'recall', cv = cv)
    gs.fit(features, labels)

    #Returns the results of the GridSearchCV
    return gs

#This function prints the results of the cross-validation
def print_results(clf):
    #Results of the best estimator investigation:
    print "Best Score:", clf.best_score_
    print ""
    print "Best Parameters:", clf.best_params_
    print ""
    
    #Checking which are the best algorithms:
    clf_results = pd.DataFrame(clf.cv_results_)
    #Select only algorithms that resulted in recall over 0.3
    clf_options = clf_results[clf_results['mean_test_recall'] >= 0.3]
    print "Estimators that resulted in recall over 0.3: \n" 
    print clf_options['param_estimator'].unique()
    print ""
    print "Recall, precision, accuracy and number of PCs for the best results: \n"
    print clf_options[['mean_test_recall', 'mean_test_precision', 
                       'mean_test_accuracy', 'param_estimator', 
                       'param_reduce_dimension__n_components']]

#This function was created to compare different types of classification 
#algorithm. It applies PCA to the data, finds the best number of PCs and
#the best classification algorithm
def compare_classifiers(pipe, features, labels):

    #Parameters to test in the cross validation process:     
    param_grid_estimators = [{'reduce_dimension__n_components': [3, 5, 7, 9, 11],
                             'estimator': [SVC(kernel = 'rbf', C = 4000, gamma = 0.01),
                                 tree.DecisionTreeClassifier(min_samples_split = 20),
                                 GaussianNB(),
                                 KNeighborsClassifier(n_neighbors = 5),
                                 AdaBoostClassifier(n_estimators = 150),
                                 RandomForestClassifier(n_estimators = 100, 
                                                        min_samples_split = 10)]},
                            {'reduce_dimension': [SelectPercentile()],
                             'reduce_dimension__percentile': [50, 60, 70],
                             'estimator': [SVC(kernel = 'rbf', C = 4000, gamma = 0.01),
                                 tree.DecisionTreeClassifier(min_samples_split = 20),
                                 GaussianNB(),
                                 KNeighborsClassifier(n_neighbors = 5),
                                 AdaBoostClassifier(n_estimators = 150),
                                 RandomForestClassifier(n_estimators = 100, 
                                                        min_samples_split = 10)]}]
                                                          
    #Apply cross validation:
    gs = kfold_cross_validation(pipe, param_grid_estimators, features, labels)
    
    #Printing the main results:
    print_results(gs)
        
    return gs

#This function does a K-fold cross-validation of the three best algorithms
#And tune the parameters in order to find the best classifier and configuration
#for this problem
def tune_classifier(pipe, features, labels):    
    #Parameters to tune the algorithms:
    #For SVC:
    C_values = [500, 1e3, 3e3, 4e3, 5e3, 1e4]
    gamma_values = [0.001, 0.01, 0.1]
    #For Adaboost:
    n_estimators=[100, 150, 200]
    #Components for PCA:
    components = [6, 7, 8, 9, 10, 11]
    
    param_grid = [{'reduce_dimension__n_components': components,
                'estimator': [SVC(kernel = 'rbf')],
                'estimator__C': C_values,
                'estimator__gamma': gamma_values
                   },
                {
                'reduce_dimension__n_components': components,
                'estimator': [AdaBoostClassifier()],
                'estimator__n_estimators': n_estimators
                   }]
    
    #Cross validation
    gs = kfold_cross_validation(pipe, param_grid, features, labels)
    
    #Results:
    print_results(gs)
    
    clf = gs.best_estimator_
    #Returns the best estimator and the GridSearch results
    return clf, gs

#Function to add the new feature to the dataset:
def test_new_feature(data):
    for name in data.keys():     
        data[name]['extras_over_salary'] = new_feature(data[name])
    return data


##################################################
#Loading and editing the data
##################################################

## Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#Removing outliers from the dataset:
data_dict.pop('TOTAL', 0 )
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0 )
data_dict.pop('BHATNAGAR SANJAY', 0 )

#Features:
features_list = ['poi']
features_name = data_dict[data_dict.keys()[0]].keys()
features_name.remove('poi')
features_name.remove('email_address')
###Features with too many missing values were removed.
features_name.remove('deferral_payments')
features_name.remove('deferred_income')
features_name.remove('director_fees')
features_name.remove('loan_advances')
features_name.remove('restricted_stock_deferred')

features_list = features_list + features_name

## Store to my_dataset for easy export below.
my_dataset = data_dict

## Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Pipeline for the cross-validation:
pipe = Pipeline([('scaling', preprocessing.RobustScaler()),
                 ('reduce_dimension', PCA()), 
                 ('estimator', SVC(kernel = 'rbf'))])
        

########################################################   
#Exploring the data and the correlation between variables
########################################################
### Uncomment the following line to see the data exploration:  
#explore_data()


#######################################################
#Cross-Validation - testing different configurations
#######################################################
#The cross-validation tunes the classification algorithm and PCA.

### Uncomment the line below to see the first cross validation applied, where 
### different algorithms were tested:
#clf_estimators = compare_classifiers(pipe, features, labels)


#######################################################
#Cross-Validation - tuning the algorithmn
#######################################################
### Second cross validation where the best algorithms were tuned.
#Comment next lines if not building the algorithm
clf, gs = tune_classifier(pipe, features, labels)

#Result of the PCA:

pca_result = clf.named_steps['reduce_dimension']

print "Percentage of variance explained by each component: \n"
print pca_result.explained_variance_ratio_
print ""
print "PCA components:"
print pca_result.components_

#Saving the results:
clf_results = pd.DataFrame(gs.cv_results_)

# Dump your classifier, dataset, and features_list:
#dump_classifier_and_data(clf, my_dataset, features_list)


#####################################################
#Testing the effect of the new feature on the final
#algorithm performance
#####################################################
### Uncomment all the lines below if testing the new feature:

## Add the new feature to the dataset and split the data into labels and features again
#data_dict = test_new_feature(data_dict)
#features_list.append('extras_over_salary')
#
#data = featureFormat(data_dict, features_list, sort_keys = True)
#labels_test, features_test = targetFeatureSplit(data)
#
##Fit the algorithm using the data with the new feature
#clf_test = gs.fit(features_test, labels_test)
#
#print "Best score:", clf_test.best_score_
#print "Parameters of the estimator:", clf_test.best_params_
#pca_result = clf_test.best_estimator_.named_steps['reduce_dimension']
#print "Percentage of variance explained by each component: \n"
#print pca_result.explained_variance_ratio_
