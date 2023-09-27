import numpy as np

from FairFIS.base import fis_score
from FairFIS.util import *
from FairFIS.fis_tree import fis_tree
from joblib import Parallel, delayed


"""
A class to modify sklearn Gradient Boosted Trees to calculate
FairFIS.
    ----------
    fitted_clf: Classifier or Regressor
       The selector is a trained standard sklearn Gradient Boosted Trees
    train_x: nDarray of shape nxm
        The dataset for training
    train_y: nDarray of shape nx1:
        The true training labels
    protected_attribute: ndarray of shape nX1
        The protected feature
    protected_value: int, Default = 0
        Protected value of the protected attribute
    normalize: bool, Default = True
        Returns normalized FairFIS score if value is set to True
    regression: bool, Default = False
        Value is True is fitted_clf is a Regressor
    multiclass: bool, Default = False
        The value is set to True for Multiclass Classification
    triangle: bool, Default = False
        True when triangle inequality is used

    -----------
    Examples
    --------
    >>> from FIS import fis_forest
    >>> clf = GradientBoostingClassifier()
    >>> clf.fit(train_x, train_y)
    >>> f_forest = fis_forest(clf,train_x,train_y,z,0)
    >>> f_forest.calculate_fairness_importance_score()
    >>> fis_dp = f_forest._fairness_importance_score_dp_root
    >>> fis_eqop = f_forest._fairness_importance_score_eqop_root
"""


class fis_boosting(fis_score):
    def __init__(self, fitted_clf,train_x,train_y, protected_attribute, protected_value, normalize = True, regression = False, multiclass = False,triangle = False):
        self.fitted_clf = fitted_clf
        self.train_x = train_x
        self.train_y = train_y
        self.protected_attribute = protected_attribute
        self.protected_value = protected_value
        self.number_of_features = train_x.shape[1]
        self._fairness_importance_score_dp = np.zeros(self.number_of_features)
        self._fairness_importance_score_eqop = np.zeros(self.number_of_features)
        self._fairness_importance_score_dp_root = np.zeros(self.number_of_features)
        self._fairness_importance_score_eqop_root = np.zeros(self.number_of_features)
        self.individual_feature_values = {}
        self.normalize = normalize
        self.regression = regression
        self.multiclass = multiclass
        self.triangle = True


    def each_tree(self,index):
        individual_tree = fis_tree(self.fitted_clf.estimators_[index,0], self.train_x, self.train_y, self.protected_attribute, self.protected_value, normalize = False, regression=self.regression, multiclass = self.multiclass, triangle=self.triangle)
        individual_tree.calculate_fairness_importance_score()
        return individual_tree
    
    def calculate_fairness_importance_score(self):
        [self.individual_feature_values.setdefault(i, []) for i in range(self.number_of_features)]
        self.trees = []
        self.train_x_with_protected = np.concatenate((self.train_x,np.reshape(self.protected_attribute,(-1,1))),axis=1) 
        
        
        self.trees = Parallel(n_jobs=-2,verbose=1)(
        delayed(self.each_tree)(index) 
        for index in range(self.fitted_clf.n_estimators_)
        )
        
        for individual_tree in self.trees:
            
            for i in range(self.number_of_features):
                self._fairness_importance_score_dp[i] += individual_tree._fairness_importance_score_dp[i]
                self._fairness_importance_score_eqop[i] += individual_tree._fairness_importance_score_eqop[i]
                self._fairness_importance_score_dp_root[i] += individual_tree._fairness_importance_score_dp_root[i]
                self._fairness_importance_score_eqop_root[i] += individual_tree._fairness_importance_score_eqop_root[i]
        self._fairness_importance_score_dp /= (self.fitted_clf.n_estimators_)
        self._fairness_importance_score_eqop /= (self.fitted_clf.n_estimators_)
        if self.normalize == True:
            self._fairness_importance_score_dp /= np.sum(abs(self._fairness_importance_score_dp))
            self._fairness_importance_score_eqop /= np.sum(abs(self._fairness_importance_score_eqop))
            self._fairness_importance_score_dp_root /= np.sum(abs(self._fairness_importance_score_dp_root))
            self._fairness_importance_score_eqop_root /= np.sum(abs(self._fairness_importance_score_eqop_root))

    