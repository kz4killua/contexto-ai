from collections import OrderedDict

from utils import get_rank


class Game:

    def __init__(self, number):
        self.number = number
        self.history = OrderedDict()

    def make_guess(self, word):
        if word not in self.history:
            self.history[word] = self.get_rank(word)
        return self.history[word]
    
    def get_rank(self, word):
        return get_rank(word, self.number)
    
    @property
    def number_of_guesses(self):
        return len(set(self.history.values()) - {None})

    @property
    def won(self):
        return 0 in self.history.values()