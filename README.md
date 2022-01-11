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
- **Epoch 0**: untrained (random move) bot
- **Epoch 1**: 10000 games, Adam(learning_rate=0.00000500, beta_1=0.90, beta_2=0.999, clipvalue=0.1), weights: (1.0, 1.0)
- **Epoch 2**: 10000 games, Adam(learning_rate=0.00000100, beta_1=0.90, beta_2=0.999, clipvalue=0.1), weights: (0.1, 1.0)
- **Epoch 3**: 10000 games, Adam(learning_rate=0.00000002, beta_1=0.90, beta_2=0.999, clipvalue=0.1), weights: (1.0, 1.0)
- **Epoch 4**: 0 games, Adam(learning_rate=0.0000008, beta_1=0.90, beta_2=0.999, clipvalue=0.1), weights: (1.0, 1.0)
- **Epoch 5**: 0 games, Adam(learning_rate=0.0000008, beta_1=0.90, beta_2=0.999, clipvalue=0.1), weights: (1.0, 1.0)
- **Epoch 6**: 0 games, Adam(learning_rate=0.0000008, beta_1=0.90, beta_2=0.999, clipvalue=0.1), weights: (1.0, 1.0)
- **Epoch 7**: 0 games, Adam(learning_rate=0.0000008, beta_1=0.90, beta_2=0.999, clipvalue=0.1), weights: (1.0, 1.0)
- **Epoch 8**: 0 games, Adam(learning_rate=0.0000008, beta_1=0.90, beta_2=0.999, clipvalue=0.1), weights: (1.0, 1.0)
- **Epoch 9**: 0 games, Adam(learning_rate=0.0000008, beta_1=0.90, beta_2=0.999, clipvalue=0.1), weights: (1.0, 1.0)
- **Epoch 10-19**: 0 games, Adam(learning_rate=0.0000008, beta_1=0.90, beta_2=0.999, clipvalue=0.1), weights: (1.0, 1.0)
- **Epoch 20-26**: 0 games, Adam(learning_rate=0.0000008, beta_1=0.90, beta_2=0.999, clipvalue=0.1), weights: (1.0, 1.0)
- **Epoch 27-29**: 0 games, Adam(learning_rate=0.0000008, beta_1=0.90, beta_2=0.999, clipvalue=0.1), weights: (1.0, 1.0)
- **Epoch 13**: 0 games, Adam(learning_rate=0.00, beta_1=0.9, beta_2=0.999, clipvalue=0.11), weights: (1.0, 1.0)
- **Epoch 14**: 0 games, Adamx, weights: (0.00, 1.0)
- ...

| Epoch |Wins | Losses | Ties |
| --- | --- | --- |
|  1 | .712 | .286 | .002 |
|  2 | .602 | .390 | .008 |
|  3 | .000 | .000 | .000 |
|  4 | .000 | .000 | .000 |
|  5 | .000 | .000 | .000 |
|  6 | .000 | .000 | .000 |
|  7 | .000 | .000 | .000 |
|  8 | .000 | .000 | .000 |
|  9 | .000 | .000 | .000 |
| 10 | .000 | .000 | .000 |

// For random moves, first player wins 53.1% of time.
// Bot 11 wins 53.7% when doing first, bot 12 wins 53.1% when doing first

Benchmarks:
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

| Plane Description |
| --- |
| all 1 if current player is Player 2 |
| 1 if current player has peg at location|
| 1 if opponent has peg at location |
| 1 if spot is legal, open |
| 1 if spot completes current player canoe |
| 1 if spot completes oppoonent canoe |
| 1 if spot is in current_player canoe |
| 1 if spot is in oppoonent canoe |
potential new planes: wins the game for self/opponent;
