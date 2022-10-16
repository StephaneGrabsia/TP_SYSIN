# TP_SYSIN - k-armed bandit problem, comparisons of different methods

This repository is an individual exercice for the Intelligent System course.
The goal is to implement the k-armed bandit problem and to compare some Reinforcement Learning strategies. 

You will find in the data folder the plots made to compare these methods. 
We can see here that the Upper Confidence Bounds performs a bit better that the epsilon-greedy with these parameters


## Installation:
The only libraries used here are numpy, random and matplotlib.pyplot. As random is in the standard library, you only have to download numpy and matplolib in your environment:

```bash
pip install numpy
pip install matplotlib
```

## Usage:
The file main.py defines the function you may want to use. All of them only have optional parameters, so if you want to produce the same graphics than those who are presented 
in the results folder, you can just uncomment the calls of these function at the bottom of the file and execute it. 

You can easily test others parameters (especially for c and eps) by just adding these parameters in the function calls. 

The results have been produced with the use of 10 gaussian distribution, defined in the file machines.py. You can also easily modify the number or parameters of these distributions. 

