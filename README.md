## Statement
This learning-project draws heavily from *Deep Learning and the Game of Go* (Manning, 2019), adapted to Dale Walton's abstract strategy game *Canoe* ([BGG page](https://boardgamegeek.com/boardgame/10571/canoe), [Elm implementation](https://github.com/bored-games/canoe-game)). The goal is to use reinforcement learning through self-play to improve a Canoe bot that starts with basically no knowledge of the rules.

## Current status
On hold. After 4 iterations, the actor-critic bot beats the random bot about 85% of the time. However, it is still trivial for a human to win against the bot. Further progression would take many more iterations, and AWS refused my request to pay for a GPU. The free tier CPU-based machine is slower than my desktop, which takes hours for each set of self-play games.

## Steps and strategies
1. Implement logic
    * Legal moves, victory conditions, agents: Random, Neighbor, and Greedy
2. Simple neural net to learn agent strategies (`nn-`)
    * Infer the rules of Canoe, roughly copying the Greedy strategy but never improving beyond it
3. Policy gradient learning (`rl-`)
    * Use self-play data to increase probability of every move that happened during a win
4. Q-learning (`q-`)
    * Input the game-state and a proposed move to estimate an action-value function which estimates the chances of winning after a particular move
5. Actor-critic learning (`ac-`)
    * Directly learn both a policy function (i.e. which move to make) and a value function (weighting the importance of each move)

## To run
1. `python ac-init-agent.py <model-output-file>`
2. `python ac-self-play.py <model-input-file> <data-output-file> --num-games 5000`
3. `python ac-train.py <model-input-file> <model-output-file> <data-input-file(s)>`
4. Repeat steps 2 & 3, evaluating progress


## Classical bot comparison (win rates)
- **Random**: choose from any open space
- **Neighbor**: choose adjacent to opponent's previous move
- **Greedy**: take any move that wins, avoid any move that allows immediately loss, else choose adjacent to opponent (note: does not automatically *block* opponent's winning canoes)

|Bot 1 | Bot 2 | Ties |
| --- | --- | --- |
| Neighbor | Random | |
| .549 | .400 | .051 |
| Greedy | Random | |
| .824 | .146 | .030 |
| Greedy | Neighbor | |
| .710 | .152 | .138 |

## ML bot comparison

- **Epoch 0**: untrained (random move) bot
- **Epoch 1**: actor-critic bot trained with 3000 self-play games using Epoch 0, SGD learning_rate=0.005
- **Epoch 2**: SGD learning_rate decreased to 0.001
- **Epoch 3**: switched optimizer to adadelta, 9000 trials
- **Epoch 4**: Use 50,000 trials
- ...

|Bot 1 | Bot 2 | Ties |
| --- | --- | --- |
| Epoch 1 | Epoch 0 | |
| .63 | .36 | .01 |
| Epoch 2 | Epoch 1 | |
| .64 | .35 | .01 |
| Epoch 3 | Epoch 2 | |
| .70 | .29 | .01 |
| Epoch 4 | Epoch 3 | |
| .66 | .33 | .01 |
| ... | Epoch 0 | |

Benchmarks:
|Bot 1 | Bot 2 | Ties |
| --- | --- | --- |
| Epoch 3 | Random | |
| .838 | .156 | .006 |

## The model
AlphaGo uses some 48 feature planes. Here we use 6 arrays, each 6-by-13:

| Plane Description |
| --- |
| all 1 if current player is Player 2 |
| 1 if current player has peg at location|
| 1 if opponent has peg at location |
| 1 if spot is legal, open |
| 1 if spot completes current player canoe |
| 1 if spot completes oppoonent canoe |
