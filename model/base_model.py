from abc import ABC, abstractmethod

class BaseModel(ABC):

  @abstractmethod
  def build(self):
    pass

  @abstractmethod
  def fit(self, input):
    pass

  @abstractmethod
  def predict(self, input):
    pass
