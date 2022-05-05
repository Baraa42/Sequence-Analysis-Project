# Sequence-Analysis-Project

All the code relevant for evaluation was developped using Google Colab. The Readme include links for all notebooks.

## Basic: TicTacToe Agent

[Notebook](https://colab.research.google.com/drive/1SdLeOYP0CYECWpo45YVsjurPzBA1j3sR#scrollTo=I9jQ8-KY28uP)

## Advanced: Pong x Breakout

### DQN Implementation and Improvements

The two improvements implemented are Double DQN and Dueling DQN, the main notebook contain the implementation of both improvements. The other 3 notebooks are copy of the main one but modified to include only one or zero improvement for ablation studies.

### Main notebooks

[Pong_DuelingDDQN](https://colab.research.google.com/drive/1KabSuK_YIPKQ6Fd-kiG6pt_xWFn9-898) <br />
[Breakout_DuelingDQN](https://colab.research.google.com/drive/1of6IMcNNDM_Z3a2_yqarv2VZtyXuuOM0)

### Ablation study notebooks

[Pong_VanillaDQN](https://colab.research.google.com/drive/1H5oXF9V53H0k_CntttsdEYjUszWjsUps) <br />
[Pong_DDQN](https://colab.research.google.com/drive/1J9AYuypKxLEXRAxBh-i5Sr9QnMnlQLRP) <br />
[Pong_DuelingDQN](https://colab.research.google.com/drive/1y8V09rim1f5xCnB11lkISXNnN4f58FC4)

### RLLIB

As said in the report, I had many bugs with RLLIB for Atari environments whose states are frame after succeding in the first time.

[Breakout](https://colab.research.google.com/drive/1BCS4rA5gevomP4xfDJb_-7nEqQqcHpP9) <br />
[Breakout_Deterministic](https://colab.research.google.com/drive/1Mhz0VVJdVnRC-b_qlSOGDznUwq5jPmJQ#scrollTo=WChlYLblwe-0)

## Extra Task: PPO for CartPole-v0

[PPO](https://colab.research.google.com/drive/1omU_pdV5P7GGVis_cTkfyUPbgYBM5VFZ#scrollTo=hgmUDEIL34KG) <br />
[Resource](https://www.youtube.com/watch?v=HR8kQMTO8bk)
