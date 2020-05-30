from abc import ABC, abstractmethod

class BaseModel(ABC):

  def __init__(self):
    print("abstract init")

  @abstractmethod
  def fit(self, data):
    pass

  @abstractmethod
  def predict(self, data):
    pass
