"""
ac-self-play.py: repeatedly play games against current iteration of agent and
  save the history/results
"""

import re
from pathlib import Path

import click
import h5py
from canoebot import agent, encoders, utils
from canoebot.board import GameState, Player, format_colored
from canoebot.experience import ExperienceCollector, combine_experience


def simulate_game(red_player: agent.Agent, yellow_player: agent.Agent) -> GameState:
    """self-play a single game, record results"""
    moves = []
    game = GameState.new_game()
    agents = {
        Player.red: red_player,
        Player.yellow: yellow_player,
    }

    while not game.is_over():
        next_move = agents[game.current_player].select_move(game)
        moves.append(next_move)
        game = game.apply_move(next_move)
    # game.print_board()
    return game


@click.command()
@click.option(
    "-i",
    "--model-file",
    type=click.Path(dir_okay=False, writable=True),
    help="Input model file",
)
@click.option(
    "-o",
    "--experience-file",
    type=click.Path(dir_okay=False, writable=True),
    help="Output file for experience",
)
@click.option(
    "-n",
    "--num-games",
    type=click.INT,
    default=1000,
    show_default=True,
    help="Number of games to play",
)
@click.option(
    "-m",
    "--mute",
    type=click.BOOL,
    default=False,
    help="Supress messages",
)
def _main(model_file, experience_file, num_games, mute):
    if experience_file is None and (m := re.search(r"(\d+)", model_file)):
        experience_file = Path("./experience/") / f"exp{m.group(1)}.h5"
    main(Path(model_file), experience_file, num_games, mute)


def main(model_file: Path, experience_file, num_games, mute) -> None:
    """self-play many games, save results"""
    agent1 = agent.ACAgent(utils.load_model(model_file), encoders.ExperimentalEncoder())
    agent2 = agent.ACAgent(utils.load_model(model_file), encoders.ExperimentalEncoder())
    # agent2 = agent.GreedyAgent(encoders.ExperimentalEncoder())
    collector1 = ExperienceCollector()
    collector2 = ExperienceCollector()
    agent1.set_collector(collector1)
    agent2.set_collector(collector2)

    first_player_wins = 0
    for i in range(num_games):
        collector1.begin_episode()
        collector2.begin_episode()

        game_record = simulate_game(agent1, agent2)  # won with {game.winning_canoes}

        if game_record.winner == Player.red:
            first_player_wins += 1
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=-1)
        elif game_record.winner == Player.yellow:
            collector1.complete_episode(reward=-1)
            collector2.complete_episode(reward=1)
        else:
            collector1.complete_episode(reward=0)
            collector2.complete_episode(reward=0)

        if not mute:
            if (i % 1000) == 0:
                experience = combine_experience([collector1, collector2])
                with h5py.File(experience_file, "w") as experience_filef:
                    experience.serialize(experience_filef)
                if not mute:
                    print(f"Saved experience for {i} games to `{experience_file}`.")

            print(
                f"{format_colored(game_record.winner, '■')}",
                end=" ",
                flush=True,
            )
            if (i + 1) % 80 == 0:
                print(f" {100*(i+1)/num_games:4.1f}%")

    experience = combine_experience([collector1, collector2])
    with h5py.File(experience_file, "w") as experience_filef:
        experience.serialize(experience_filef)
    print("")
    print(f"Saved experience for {i+1} games to `{experience_file}`.")
    print(f"First player wins: {first_player_wins / num_games}")


if __name__ == "__main__":
    _main()
