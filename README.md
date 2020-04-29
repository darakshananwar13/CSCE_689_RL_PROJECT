# Reinfrocement-Learning-Project
## All Goal Update RL using CNN

- This repositary contains the code for our project in CSCE 689 Reinforcement Learning taught in Spring 
2020 by [Prof. Guni Sharon](http://faculty.cse.tamu.edu/guni/).
- Algorithm and results presented here are based on the paper 
[Scaling All-Goals Updates in Reinforcement Learning Using Convolutional Neural Networks(https://arxiv.org/abs/1810.02927).

## Steps to follow for running the given source code 

1. Install  TensorFlow, Baselines, Gym and Gym Retro.<br />
   Please find the following command used for installing these in google colab.<br />
   
    ! git clone https://github.com/openai/baselines.git <br />
    % cd baselines <br />
    ! pip install -e . <br />
    % cd .. <br />
    !apt-get install pkg-config lua5.1 build-essential libav-tools git <br />
    !pip install tqdm retrowrapper gym-retro <br />
    !pip install -U git+git://github.com/frenchie4111/dumbrain.git <br />
    %tensorflow_version 1.x <br />


2. Clone our repository and install the setup-tools dependencies using pip install -e command. <br />
   Please find the following commands for the same. <br />

    ! git clone https://github.com/darakshananwar13/CSCE_689_RL_PROJECT.git <br />
    % cd CSCE_689_RL_PROJECT/qmap-master/ <br />
    !pip install -e . <br />



3. Run train_gridworld.py in order to train for grid world. <br />
4. Run train_montezuma.py in order to train model for montezume game. <br />
5. In order to run Super Mario game, we need to copy the game environment in the directory where GYM retro is installed. <br /> 
   Please follow the commands below to copy in google-colab environment. <br />
   
    !cp "/content/CSCE_689_RL_PROJECT/qmap-master/SuperMarioAllStars-Snes.zip" "/usr/local/lib/python3.6/dit-  packages/retro/data/stable" <br />
    % cd /usr/local/lib/python3.6/dist-packages/retro/data/stable <br />
    !unzip SuperMarioAllStars-Snes.zip <br />

   Now, run train_mario.py for training the model for Mario game. <br />
   
6. We can find the videos of the game played during training saved under “result” folder created. <br />


