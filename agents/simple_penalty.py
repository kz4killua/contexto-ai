import numpy as np

from game import Game
from utils import load_words, get_distances


class SimplePenaltyAgent:

    def __init__(self, game: Game):
        self.game = game
        self.words, self.embeddings = load_words()
        self.scores = np.zeros(len(self.words), dtype=np.float64)

    def play(self, log=False):
        
        while True:

            guess = self.get_best_guess()
            rank = self.game.make_guess(self.words[guess])

            if log:
                print(f"Guess #{self.game.number_of_guesses}: {self.words[guess]}, Rank: {rank}")

            # Skip invalid words
            if rank is None:
                self.scores[guess] = np.inf
                continue

            # Skip guesses already made
            if rank in list(self.game.history.values())[:-1]:
                self.scores[guess] = np.inf
                continue

            if rank == 0:
                return self.words[guess]
            
            self.update_scores(guess, rank)


    def update_scores(self, last_guess: int, last_rank: int):

        for word, rank in self.game.history.items():

            if rank is None:
                continue
            if word == self.words[last_guess]:
                continue
            if rank == last_rank:
                continue

            w1, r1 = word, rank
            w2, r2 = self.words[last_guess], last_rank

            d1 = get_distances(self.embeddings, self.embeddings[self.words == w1])
            d2 = get_distances(self.embeddings, self.embeddings[self.words == w2])

            if r1 < r2:
                self.scores += (d1 > d2).astype(int)
            if r1 > r2:
                self.scores += (d1 < d2).astype(int)

        self.scores[last_guess] = np.inf

    def get_best_guess(self) -> int:
        """Returns the index of the word with the best (lowest) score."""
        return np.argmin(self.scores)