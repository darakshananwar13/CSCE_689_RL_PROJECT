# Reinfrocement-Learning-Project
## All Goal Update RL using CNN

- This repositary contains the code for our project in CSCE 689 Reinforcement Learning taught in Spring 
2020 by [Prof. Guni Sharon](http://faculty.cse.tamu.edu/guni/).
- Algorithm and results presented here are based on the paper 
[Scaling All-Goals Updates in Reinforcement Learning Using Convolutional Neural Networks(https://arxiv.org/abs/1810.02927).

## Steps to follow for running the given source code 

1. Install  TensorFlow, Baselines, Gym and Gym Retro.
   Please find the following command used for installing these in google colab.
    ! git clone https://github.com/openai/baselines.git
    % cd baselines
    ! pip install -e .
    % cd ..
    !apt-get install pkg-config lua5.1 build-essential libav-tools git
    !pip install tqdm retrowrapper gym-retro
    !pip install -U git+git://github.com/frenchie4111/dumbrain.git
    %tensorflow_version 1.x


2. Clone our repository and install the setup-tools dependencies using pip install -e command.
   Please find the following commands for the same.

    ! git clone https://github.com/darakshananwar13/CSCE_689_RL_PROJECT.git
    % cd CSCE_689_RL_PROJECT/qmap-master/
    !pip install -e .



3. Run train_gridworld.py in order to train for grid world. 
4. Run train_montezuma.py in order to train model for montezume game.
5. In order to run Super Mario game, we need to copy the game environment in the directory where GYM retro is installed. 
   Please follow the commands below to copy in google-colab environment.
    !cp "/content/CSCE_689_RL_PROJECT/qmap-master/SuperMarioAllStars-Snes.zip" "/usr/local/lib/python3.6/dit-  packages/retro/data/stable"
    % cd /usr/local/lib/python3.6/dist-packages/retro/data/stable
    !unzip SuperMarioAllStars-Snes.zip

   Now, run train_mario.py for training the model for Mario game.
6. We can find the videos of the game played during training saved under “result” folder created.


