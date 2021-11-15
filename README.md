- **Random**: choose from any open space
- **Greedy**: choose adjacent to opponent's previous move
- **Greedy greedy**: take any move that wins, avoid any move that immediately loses, else choose adjacent to opponent (note: does not automatically block winning canoes)

## Bot comparison

|Bot 1 | Bot 2 | Ties |
| --- | --- | --- |
| Greedy | Random | |
| .58 | .36 | .06 |
| Greedy | Greedy | |
| .43 | .40 | .17 |
| Greedy greedy | Greedy | |
| .79 | .13 | .08 |