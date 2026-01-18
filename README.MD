This is a clone of NanoGPT created by Andrej Karpathy: https://github.com/karpathy/nanoGPT

This codebase is my attempt to delve deep into the world of Neural Networks and Deep Learning, following the video series meticulously created and gifted by Andrej Karpathy on youtube: https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ

## File structure

```
.
├── LICENSE
├── README.md
├── config
├── dev                                     # this folder is for scribbling, brain storming and figuring out stuff
│   ├── bigram.py                           # Simple bigram model
│   ├── gpt_dev.ipynb                       # building the GPT layers
│   ├── play.ipynb                          # more scribbling....
│   ├── train_gpt2_DDP.py                   # Exploring DDP, training on edu_fineweb10B and evaluating hellaswag
│   └── train_gpt2_simple_edu_fineweb.py    # training on edu_fineweb10B without DDP
│   └── train_gpt2_simple_shakespeare.py    # training on shakespeare dataset without DDP
├── data                                    # folder contains scripts for preparation of training data
├── imgs                                    # some imags for reference
├── papers                                  # Papers I referenced during the development
├── model.py                                # NanoGPT model
├── sample.py                               # Sample from the model based on the training data
├── scaling_laws.ipynb                      # Understanding the scaling laws - have to spend more time... :(
├── train.py                                # Script for training the model with any dataset
├── transformer_sizing                      # Analysis of transformers and flop -  have to spend more time... :(
```
