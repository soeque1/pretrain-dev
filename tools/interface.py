from abc import ABCMeta, abstractmethod


class MyProgram(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def sampling(self, text: List[str]) -> List[str]:
        raise NotImplementedError('')

    @abstractmethod
    def pre_tokenizer(self, new_tokens: List[str]) -> int:
        raise NotImplementedError('')

    @abstractmethod
    def train(self, files: List[str]) -> none:
        raise NotImplementedError('')


class PreTokenizer(metaclass=ABCMeta):
    def __init__(self):
        pass


