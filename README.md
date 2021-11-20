## Statement
This learning-project draws heavily from *Deep Learning and the Game of Go* (Manning, 2019), adapted to Dale Walton's abstract strategy game *Canoe* ([BGG page](https://boardgamegeek.com/boardgame/10571/canoe), [Elm implementation](https://github.com/bored-games/canoe-game)). The goal is to use reinforcement learning through self-play to improve a Canoe bot.

### Steps and strategies
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

## Classical bot comparison (win rates)

- **Random**: choose from any open space
- **Neighbor**: choose adjacent to opponent's previous move
- **Greedy**: take any move that wins, avoid any move that allows immediately loss, else choose adjacent to opponent (note: does not automatically *block* winning canoes)

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
- **Epoch 1**: actor-critic bot trained with 3000 self-play games using Epoch 0, learning_rate=0.005
- **Epoch 2**: lr = 0.00005
- ...
- **Epoch n**: actor-critic bot trained with *n* iterations of 4500 self-play games (each Epoch n-1)

|Bot 1 | Bot 2 | Ties |
| --- | --- | --- |
| Epoch 1 | Epoch 0 | |
| .630 | .360 | .010 |
| Epoch 2 | Epoch 1 | |
| .710 | .280 | .010 |
| Epoch 3 | Epoch 2 | |
| ... | ... | ... |
| Epoch 3 | Epoch 0 | |
| - | - | - |
| Epoch 10 | Epoch 0 | |
| - | - | - |
| Epoch 10 | Greedy | |
| - | - | - |