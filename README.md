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
However, please note that the results are not final and are subject to change. STake for example the Car-Racing agent. I was able to train it to a certain level of performance, but I am still working on improving it.
<figure class="video_container">
  <iframe src="discrete/video/evaluation_51.mp4" frameborder="0" allowfullscreen="true"> 
</iframe>
</figure>
[evaluation_51.mp4](discrete%2Fvideo%2Fevaluation_51.mp4)

Here is an example of the Car-Racing agent after 51 episodes of training. The agent is able to drive around the track, but it is not able to complete a full lap. The agent is still learning and I am working on improving it.

### Structure
 I am building modular code that is easy to understand and modify. I am using several abstract classes to define the structure of the algorithms and environments. 
 This helps me in abstracting away the logic of the agent from the main function. So the main function just needs to call the abstract method learn()  and the agent will learn from the environment. This makes the code more readable and easier to modify.