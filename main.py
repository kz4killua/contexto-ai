import time
import requests
import numpy as np
from numpy.linalg import norm

from dictionary import dictionary


MODEL = "words.txt"


def load_words(path):
    """Loads and returns words and their embeddings."""

    words = list()
    embeddings = list()

    with open(path, encoding="utf-8") as f:
        for line in f.readlines():
            line = line.split()
            word = line[0]
            embedding = [float(x) for x in line[1:]]

            # Filter out non-alphabetic, non-ASCII words
            if not (word.isalpha() and word.isascii()):
                continue
            # Filter out invalid English words
            if word not in dictionary:
                continue

            words.append(word)
            embeddings.append(embedding)

    # Store embeddings in numpy array
    embeddings = np.array(embeddings)

    return words, embeddings


def get_distance(embeddings, guess):
    """Returns the distances between the word embeddings and a guess."""
    # return 1 - (np.dot(embeddings, guess) / (norm(embeddings, axis=1) * norm(guess)))
    return norm(embeddings - guess, axis=1)


def make_guess(scores: np.ndarray) -> int:
    """Returns the index of the word with the best (lowest) score"""
    return np.argmin(scores)


def get_rank(word, game) -> (int | None):
    """Returns the rank of a word in a Contexto game."""
    response = requests.get(
        f"https://api.contexto.me/machado/en/game/{game}/{word}"
        ).json()
    
    if "error" in response:
        rank = None 
    else:
        rank = response["distance"]

    return rank


def solve(game, words, embeddings, log=False):
    """Solves a Contexto game."""

    # Each word will be assigned a score.
    scores = np.zeros(len(words), dtype=np.float64)

    previous_guess = None
    previous_rank = None

    explored = set()

    n_guesses = 0

    for i in range(len(words)):
        current_guess = make_guess(scores)
        current_rank = get_rank(words[current_guess], game)

        # Skip invalid words and guesses already made
        if (current_rank == None) or (current_rank in explored):
            scores[current_guess] = np.inf
            continue
        else:
            explored.add(current_rank)
            n_guesses += 1

        if log:
            print(f"Guess #{n_guesses}: {words[current_guess]}; Distance: {current_rank}")

        # Print the solution once it is found.
        if current_rank == 0:
            print(f"Solution: {words[current_guess]}\n")
            return words[current_guess], n_guesses

        if previous_guess != None:

            # Compare the last two guesses
            if current_rank < previous_rank:
                better_guess = current_guess
                worse_guess = previous_guess
            else:
                better_guess = previous_guess
                worse_guess = current_guess

            # Check the distance between each embedding and both guesses
            distance_from_better = get_distance(embeddings, embeddings[better_guess])
            distance_from_worse = get_distance(embeddings, embeddings[worse_guess])

            # Subtract and update scores
            s = distance_from_better - distance_from_worse
            scores += s

        # Update the score of the current guess
        scores[current_guess] = np.inf

        # Update previous guess and rank
        previous_guess = current_guess
        previous_rank = current_rank



def main():

    # Load words and embeddings from the model.
    print("Loading words...")
    words, embeddings = load_words(MODEL)
    print("Finished loading words.")

    for game in range(10):
        print(f"Playing game #{game}...")
        solve(game, words, embeddings)


if __name__ == "__main__":
    main()