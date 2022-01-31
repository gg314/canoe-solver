"""
ac-self-play.py: repeatedly play games against current iteration of agent and
  save the history/results
"""

import argparse
import h5py
from canoebot import agent
from canoebot import encoders
from canoebot import utils
from canoebot.board import Player, GameState, format_colored
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


def main() -> None:
    """self-play many games, save results"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-in", required=True)
    parser.add_argument("--experience-out")
    parser.add_argument("--num-games", "-n", type=int, default=10000)
    parser.add_argument("-m", "--mute", help="suppress messages", action="store_true")
    args = parser.parse_args()
    model_in_filename = "ac-v" + str(args.model_in)
    experience_filename = (
        args.experience_out if args.experience_out else f"exp{args.model_in}-1"
    )
    num_games = args.num_games

    agent1 = agent.ACAgent(
        utils.load_model(model_in_filename), encoders.ExperimentalEncoder()
    )
    agent2 = agent.ACAgent(
        utils.load_model(model_in_filename), encoders.ExperimentalEncoder()
    )
    collector1 = ExperienceCollector()
    collector2 = ExperienceCollector()
    agent1.set_collector(collector1)
    agent2.set_collector(collector2)

    first_player_wins = 0
    for i in range(args.num_games):
        collector1.begin_episode()
        collector2.begin_episode()

        game_record = simulate_game(agent1, agent2)  # won with {game.winning_canoes}
        if not args.mute:
            print(f"Game {i+1}/{num_games} {format_colored(game_record.winner, '■')}")
        if game_record.winner == Player.red:
            collector1.complete_episode(reward=1)
            collector2.complete_episode(reward=-1)
        elif game_record.winner == Player.yellow:
            first_player_wins += 1
            collector1.complete_episode(reward=-1)
            collector2.complete_episode(reward=1)
        else:
            collector1.complete_episode(reward=0)
            collector2.complete_episode(reward=0)

        if ((i - 1) % 1000) == 0:
            experience = combine_experience([collector1, collector2])
            with h5py.File(
                "./generated_experience/" + experience_filename + ".h5", "w"
            ) as experience_outf:
                experience.serialize(experience_outf)
            if not args.mute:
                print(f"Saved experience for {i-1} games.")

    experience = combine_experience([collector1, collector2])
    with h5py.File(
        "./generated_experience/" + experience_filename + ".h5", "w"
    ) as experience_outf:
        experience.serialize(experience_outf)
    print(f"First player wins: {first_player_wins / args.num_games}")


if __name__ == "__main__":
    main()
