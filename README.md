## Statement
This learning-project draws heavily from *Deep Learning and the Game of Go* (Manning, 2019), adapted to Dale Walton's abstract strategy game *Canoe* ([BGG page](https://boardgamegeek.com/boardgame/10571/canoe), [Elm implementation](https://github.com/bored-games/canoe-game)). The goal is to use convolutional neural nets and reinforcement learning through self-play to improve a Canoe bot that starts with basically no knowledge of the game.

In this version of Canoe, both players alternate placing pegs and have no opportunity to retract a peg once it is placed.

## Current status
Learning stalls after just a few iterations. The actor-critic bot beats the random bot 90%+ of the time. However, it is still trivial for a human to win against the bot. Further progress might take more training iterations (AWS mysteriously refused my request to pay for a GPU). Many paths forward, including: try to solve the case where a single canoe wins; embrace AlphaZero's Monte-Carlo rollouts and tree search.

## Steps and strategies
1. Implement logic
    * Legal moves, victory conditions, program simple agents: Random, Neighbor, and Greedy
2. Simple neural net to learn agent strategies (`nn-`)
    * Infer the rules of Canoe, roughly copying the Greedy strategy but never improving beyond it
3. Policy gradient learning (`rl-`)
    * Use self-play data to increase probability of every move that happened during a win
4. Q-learning (`q-`)
    * Input the game-state and a proposed move to estimate an action-value function which predicts the chance of winning after a particular move
5. Actor-critic learning (`ac-`)
    * Directly learn both a policy function (i.e. which move to make) and a value function (weighting the importance of each move)

## To run
1. `python ac-init-agent.py -o <model-output-file>`
2. `python ac-self-play.py -i <model-input-file> --o <data-output-file> --num-games 5000`
3. `python ac-train.py -i <model-input-file> -o <model-output-file> -e <data-input-file>`
4. Repeat steps 2 & 3, evaluating progress


## Classical bot agent comparison (win rates)
- **Random**: choose from any open space
- **Neighbor**: choose adjacent to opponent's previous move
- **Greedy**: take any move that wins, avoid any move that allows immediate loss, else choose adjacent to opponent (note: does not automatically *block* opponent's winning canoes)

|Bot 1 | Bot 2 | Ties |
| --- | --- | --- |
| Neighbor | Random | |
| .549 | .400 | .051 |
| Greedy | Random | |
| .824 | .146 | .030 |
| Greedy | Neighbor | |
| .710 | .152 | .138 |

For random moves, first player wins 53.1% of time.


Benchmarks (unfinished):
|Bot 1 | Bot 2 | Ties |
| --- | --- | --- |
| Epoch 8 | Random | |
| .000 | .000 | .000 |
| Epoch 9 | Random | |
| .000 | .000 | .000 |
| Epoch 10 | Random | |
| .000 | .000 | .000 |

## The encoding model
AlphaGo uses some 48 feature planes. Here we use 8 arrays, each 6-by-13:

| Plane | Description |
| --- | --- |
| 0 | all 1 if current player is Player 2 |
| 1 | 1 if current player has peg at location|
| 2 | 1 if opponent has peg at location |
| 3 | 1 if spot is legal, open |
| 4 | 1 if spot completes current player canoe |
| 5 | 1 if spot completes oppoonent canoe |
| 6 | 1 if spot is in current_player canoe |
| 7 | 1 if spot is in oppoonent canoe |
potential new planes: wins the game for self/opponent