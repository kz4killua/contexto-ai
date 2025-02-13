from agents.simple_penalty import SimplePenaltyAgent
from game import Game


def main():
    for n in [7, 8, 9]:
        game = Game(n)
        agent = SimplePenaltyAgent(game)
        agent.play(log=True)
        print(f"Solved game {n} in {agent.game.number_of_guesses}")


if __name__ == "__main__":
    main()