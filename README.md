## Deep Reinforcement Learning experiments on OpenAI Gymenvironments

### Introduction
In this environment I am experimenting with Deep Reinforcement Learning algorithms on OpenAI Gym environments. The main goal is to learn how to implement and tune the algorithms, and to understand how they work. Additionally, I want to establish best practices for training and evaluation of the models.


### Environments
The environments I am using are from the OpenAI Gym library. The environments are:
- CarRacing-v2
- ALE/Pacman-v5


### Algorithms
The algorithms I am using are:
- Deep Q-Network (DQN)
- Double Deep Q-Network (DDQN)
- Proximal Policy Optimization (PPO)
- Prioritized Experience Replay (PER)


### Results
The results of the experiments are stored in the runs and videos folders. The runs folder contains the training logs and the videos folder contains the videos of the trained agents.
However, please note that the results are not final and are subject to change. STake for example the Car-Racing agent. I was able to train it to a certain level of performance, but I am still working on improving it. Nevertheless. there are already some interesting strategies found by the agent. Currently, it is learning how get consistently back on the road, when it missed a turn. Furthermore, the agent does not understand yet to brake in sharp turns. Either way, this behavior is not surprising given the fact that the agent was only trained for about 30k steps. Good results require around one million steps which is why I keep training it right now.

https://github.com/Coluding/RL-Trial/assets/98786106/e2991311-232d-43e3-9440-ed0557fba1e9

\\

https://github.com/Coluding/RL-Trial/assets/98786106/e7540fd6-1888-4dce-abc6-bcae4475e042


Here is an example of the Car-Racing agent after around 800 episodes of training. The agent is able to drive around the track, but it is not able to complete a full lap. The agent is still learning and I am working on improving it. I cut the video just to display a couple hundred steps. This agent was trained using PPO. I tried with DDQL, however the learning was way more sensitive to hyperparameter and less robust. 

### Structure
 I am building modular code that is easy to understand and modify. I am using several abstract classes to define the structure of the algorithms and environments. 
 This helps me in abstracting away the logic of the agent from the main function. So the main function just needs to call the abstract method learn()  and the agent will learn from the environment. This makes the code more readable and easier to modify.











