import numpy as np

from game import Game
from utils import get_distances


class SimplePenaltyAgent:

    def __init__(self, game: Game, words: np.ndarray, embeddings: np.ndarray):
        self.words = words
        self.embeddings = embeddings
        self.game = game
        self.penalties = np.zeros(len(self.words), dtype=np.float64)

    def play(self, moves: int, log: bool=False):
        """Play the game for a given number of moves."""
        
        for _ in range(moves):

            guess = self.get_best_guess()
            rank = self.game.make_guess(self.words[guess])

            if log:
                print(f"Guess #{self.game.number_of_guesses}: {self.words[guess]}, Rank: {rank}")

            # Skip invalid words
            if rank is None:
                self.penalties[guess] = np.inf
                continue

            # Skip guesses already made
            if rank in list(self.game.history.values())[:-1]:
                self.penalties[guess] = np.inf
                continue

            if rank == 0:
                return self.words[guess]
            
            self.update_penalties(guess, rank)

    def update_penalties(self, last_guess: int, last_rank: int):
        """Add penalties to words that are inconsistent with the ranks so far."""

        for word, rank in self.game.history.items():

            if rank is None:
                continue
            if word == self.words[last_guess]:
                continue
            if rank == last_rank:
                continue

            w1, r1 = word, rank
            w2, r2 = self.words[last_guess], last_rank

            d1 = get_distances(self.embeddings, self.embeddings[self.words == w1][0])
            d2 = get_distances(self.embeddings, self.embeddings[self.words == w2][0])

            if r1 < r2:
                self.penalties += (d1 > d2).astype(int)
            if r1 > r2:
                self.penalties += (d1 < d2).astype(int)

        self.penalties[last_guess] = np.inf

    def get_best_guess(self) -> int:
        """Returns the index of the word with the best (lowest) penalty."""
        return np.argmin(self.penalties)