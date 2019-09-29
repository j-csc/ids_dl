# Core Libraries
import pandas as pd
import numpy as np
import os
from joblib import dump, load

# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore")
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000

# MLxtend
from mlxtend.feature_selection import ColumnSelector

# Models
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# Misc
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from scipy.spatial.distance import cdist
from sklearn.utils import resample
from sklearn.metrics import silhouette_score, homogeneity_completeness_v_measure, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score

# Main model builder
class modelBldr():
  # Variables
  numerical_cols = pd.DataFrame()
  target_variables = pd.DataFrame()
  X_train = []
  X_test = []
  y_train = []
  y_test = []

  # Init function
  def __init__(self,targetVariablesPath="./data/clean_target_vars.h5", numericalColumnsPath="./data/clean_num_cols.h5", downsample=True):
    print("Preparing variables for model...\n")
    self.numerical_cols = pd.read_hdf(numericalColumnsPath)
    self.target_variables = pd.read_hdf(targetVariablesPath)
    self.train_test()
    if downsample == True:
      self.X_train, self.y_train = self.downsample(self.X_train, self.X_test, self.y_train, self.y_test)

  def train_test(self):
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.numerical_cols, self.target_variables, test_size=0.2, random_state=42)

  def eval_metrics(self, model, y_pred):
    print('Accuracy: {}\n'.format(accuracy_score(self.y_test, y_pred)))
    print('Precision: {}\n'.format(precision_score(self.y_test, y_pred, average='weighted')))
    print('Recall: {}\n'.format(recall_score(self.y_test, y_pred, average='weighted')))
    pass

  def downsample(self, x_tr, x_te, y_tr, y_te):
    print("Downsampling dataset...\n")
    train_df = pd.concat([self.X_train, self.y_train], axis=1)
    mal = train_df[train_df.label != 0]
    ben = train_df[train_df.label == 0]
    ben_downsample = resample(ben, replace=False, n_samples=len(mal), random_state=42)
    downsampled_df = pd.concat([ben_downsample, mal])
    print("Result of downsampling...\n")
    print(downsampled_df.label.value_counts())
    return downsampled_df.drop('label', axis=1), downsampled_df.label

  def saveModel(self, model, name):
    dump(model, '{}.joblib'.format(name))

  def logistic_regression_model(self, downSample=True, randomState=42):
    X_tr = self.X_train
    y_tr = self.y_train
    # if downSample == True:
    #   X_tr, y_tr = self.downsample(self.X_train, self.X_test, self.y_train, self.y_test)

    # Fit model
    print("Fitting logistic regression model with X_train and y_train...\n")
    lr_model = LogisticRegression(random_state=randomState)
    lr_model.fit(X_tr, y_tr)
    y_pred = lr_model.predict(self.X_test)
    return lr_model, y_pred

  def id3_decision_tree_model(self, downSample=True, randomState=42):
    X_tr = self.X_train
    y_tr = self.y_train
    # if downSample == True:
    #   X_tr, y_tr = self.downsample(self.X_train, self.X_test, self.y_train, self.y_test)
    
    # Fit model
    print("Fitting id3 decision tree model with X_train and y_train...\n")
    id3_tree_clf = tree.DecisionTreeClassifier()
    id3_tree_clf.fit(X_tr, y_tr)
    y_pred = id3_tree_clf.predict(self.X_test)
    return id3_tree_clf, y_pred

  def random_forest_model(self, downSample=True, randomState=42):
    X_tr = self.X_train
    y_tr = self.y_train
    # if downSample == True:
    #   X_tr, y_tr = self.downsample(self.X_train, self.X_test, self.y_train, self.y_test)
    
    # Fit model
    print("Fitting random forest model with X_train and y_train...\n")
    rf_clf = RandomForestClassifier(random_state=randomState, max_depth=5, max_features='auto', criterion='gini', bootstrap=True)
    rf_clf.fit(X_tr, y_tr)
    y_pred = rf_clf.predict(self.X_test)
    return rf_clf, y_pred