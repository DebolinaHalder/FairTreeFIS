from abc import ABC, abstractmethod
class fis_score(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def calculate_fairness_importance_score(self):
        r"""
        Fits the tree or forest and gives fairness score.
        Parameters
        ----------
        X : ndarray
            Input data matrix.
        y : ndarray
            Output (i.e. response) data matrix.
        """
        pass
    
    