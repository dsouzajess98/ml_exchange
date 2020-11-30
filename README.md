# Secure Exchange of Machine Learning Models using Blockchain

Machine Learning algorithms are being developed and improved at an incredible rate, but are not necessarily getting more accessible to the broader community. Our model enables anyone to get access to high quality, objectively measured machine learning models. At Algorithmia, we believe that widespread access to algorithms and deployment solutions is going to be a fundamental building block of a balanced future for AI, and DanKu is a step towards that vision.

The DanKu protocol utilizes blockchain technology via smart contracts. The contract allows anyone to post a data set, an evaluation function, and a monetary reward for anyone who can provide the best trained machine learning model for the data. Participants train deep neural networks to model the data, and submit their trained networks to the blockchain. The blockchain executes these neural network models to evaluate submissions, and ensure that payment goes to the best model.



## 0. Requirements

To run the DanKu unittests, we first need to setup our local development environment. The following commands have been tested for OSX and Linux.

### 0.1. Installing the Solidity Compiler

### For OSX

Install a these homebrew packages:

```
brew install pkg-config libffi autoconf automake libtool openssl
```

Install the solidity compiler (solc):

```
brew update
brew upgrade
brew tap ethereum/ethereum
brew install solidity
brew link solidity
```

### For Linux

```
sudo add-apt-repository ppa:ethereum/ethereum
sudo apt-get update
sudo apt-get install solc libssl-dev
```

### 0.2. Initialize your Virtual Environment

Install [virtualenv](https://virtualenv.pypa.io/en/stable/) if you don't have it yet. (Comes installed with Python3.6)

Setup a virtual environment with Python 3:

```
cd danku;
python3.6 -m venv venv;
source venv/bin/activate;

```

### 0.3. Install the Populus Framework

Install `populus` and other requirements while in virtualenv:

```
pip install -r requirements.txt
```

Yay! You should be able to develop Ethereum contracts in Python 3 now!

## 1. Populus

All contracts are developed and tested using the Populus framework.

To compile all contracts, run the following:

```
populus compile
```

To run all the tests, use the following:

```
python -m pytest --disable-pytest-warnings tests/*
```

## 2. Danku Contracts

The DanKu contract can be found in the `contracts` directory.

For more information about DanKu contracts, please read the [white paper](#).


Super User username: dsouzajess password:hello123
Passwords for horton chain: account: seed 
