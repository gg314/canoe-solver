## Statement
This learning-project draws heavily from *Deep Learning and the Game of Go* (Manning, 2019), adapted to Dale Walton's abstract strategy game *Canoe* ([BGG page](https://boardgamegeek.com/boardgame/10571/canoe), [Elm implementation](https://github.com/bored-games/canoe-game)). The goal is to use convolutional neural nets and reinforcement learning through self-play to improve a Canoe bot that starts with basically no knowledge of the rules.

## Current status
On hold. After 4 iterations, the actor-critic bot beats the random bot about 85% of the time. However, it is still trivial for a human to win against the bot. Further progression would take many more iterations, and AWS refused my request to pay for a GPU. The free tier CPU-based machine is slower than my desktop, which takes hours for each set of self-play games (limited by `keras.predict()`).

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
2. `python ac-self-play.py --model-in <model-input-file> --experience-out <data-output-file> --num-games 5000`
3. `python ac-train.py --model-in <model-input-file> --model-out <model-output-file> <data-input-file(s)>`
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

## actor-critic bot comparison

Each bot is trained from the previous epoch self-play games.
- **Epoch 1**: untrained (random move) bot
- **Epoch 2**:  5000 games, Adam(learning_rate=0.00002, beta_1=0.90, beta_2=0.999, clipvalue=0.2), weights: (1.0, 1.0)
- **Epoch 3**:  5000 games, Adam(learning_rate=0.000005, beta_1=0.90, beta_2=0.999, clipvalue=0.2), weights: (0.5, 1.0)
- **Epoch 4**:  5000 games, Adam(learning_rate=0.00300, beta_1=0.90, beta_2=0.999, clipvalue=0.2), weights: (0.2, 1.0)
- **Epoch 5**: 10000 games, Adam(learning_rate=0.00070, beta_1=0.90, beta_2=0.999, clipvalue=0.2), weights: (1.0, 1.0)
- **Epoch 6**: 10000 games, Adam(learning_rate=0.00070, beta_1=0.90, beta_2=0.999, clipvalue=0.2), weights: (0.5, 1.0)
- **Epoch 7**: 10000 games, Adam(learning_rate=0.00007, beta_1=0.90, beta_2=0.999, clipvalue=0.2), weights: (0.2, 1.0)
- **Epoch 8**: 10000 games, Adam(learning_rate=0.00020, beta_1=0.90, beta_2=0.999, clipvalue=0.1), weights: (0.5, 1.0)
- **Epoch 9**: 10000 games, Adam(learning_rate=0.0000007, beta_1=0.90, beta_2=0.999, clipvalue=0.1), weights: (0.2, 1.0)
- **Epoch 10**: 10000 games, Adam(learning_rate=0.0000003, beta_1=0.90, beta_2=0.999, clipvalue=0.1), weights: (0.1, 1.0)
- **Epoch 11**: 10000 games, Adam(learning_rate=0.0000002, beta_1=0.90, beta_2=0.999, clipvalue=0.3), weights: (1.0, 1.0)
- **Epoch 12**: 120000 games, Adam(learning_rate=0., beta_1=0.90, beta_2=0.999, clipvalue=0.2), weights: (1.0, 1.0)
- **Epoch 13**: 00000000000000 games, Adam(learning_rate=0.000007, beta_1=0.9, beta_2=0.999, clipvalue=0.01), weights: (1.0, 1.0)
- **Epoch 14**: 10000 games, Adamx, weights: (0.2, 1.0)
- ...

|Bot 1 | Bot 2 | Ties |
| --- | --- | --- |
| Epoch 2 | Epoch 1 | |
| .716 | .280 | .004 |
| Epoch 3 | Epoch 2 | |
| .656 | .336 | .008 |
| Epoch 4 | Epoch 3 | |
| .560 | .428 | .012 |
| Epoch 5 | Epoch 4 | |
| .716 | .280 | .004 |
| Epoch 6 | Epoch 5 | |
| .728 | .252 | .020 |
| Epoch 7 | Epoch 6 | |
| .616 | .372 | .012 |
| Epoch 8 | Epoch 7 | |
| .757 | .180 | .063 |
| Epoch 9 | Epoch 8 | |
| .727 | .270 | .003 |
| Epoch 10 | Epoch 9 | |
| .620 | .377 | .003 |
| Epoch 11 | Epoch 10 | |
| .570 | .427 | .003 |
| Epoch 12? | Epoch 11 | |
| .520 | .474 | .006 |

For random moves, first player wins 53.1% of time.
Bot 11 wins 53.7% when doing first, bot 12 wins 53.1% when doing first

Benchmarks:
|Bot 1 | Bot 2 | Ties |
| --- | --- | --- |
| Epoch 8 | Random | |
| .888 | .108 | .004 |
| Epoch 9 | Random | |
| .918 | .080 | .002 |
| Epoch 10 | Random | |
| .926 | .070 | .004 |
| Epoch 11 | Random | |
| .934 | .064 | .002 |
| Epoch 12 | Random | |
| .934 | .064 | .002 |

## The encoding model
AlphaGo uses some 48 feature planes. Here we use 6 arrays, each 6-by-13:

| Plane Description |
| --- |
| all 1 if current player is Player 2 |
| 1 if current player has peg at location|
| 1 if opponent has peg at location |
| 1 if spot is legal, open |
| 1 if spot completes current player canoe |
| 1 if spot completes oppoonent canoe |
potential new planes: wins the game for self/opponent; spot is part of current canoe for self/opponent
