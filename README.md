# SNN-IIR
This repo contains the implementation of paper [Exploiting Neuron and Synapse Filter Dynamics in Spatial Temporal Learning of Deep Spiking Neural Network](https://www.ijcai.org/Proceedings/2020/388). It trains spiking neural network to learn spatial temporal patterns.

## Run

Just simply clone the repo. A few examples are provided:
- snn_mlp_1.py  
A multi-layer fully connected SNN to classify MNIST. It uses dual exponential PSP kernel.  
Following command trains the SNN using the configurations stored in snn_mlp_1.yaml
```
  python snn_mlp_1.py --train
```

Following command loads the pretrained model ./checkpoint/pretrained_snn_mlp_1 and and test it.  
```
  python snn_mlp_1.py --test
```

- snn_mlp_2.py  
A multi-layer fully connected SNN to classify MNIST. It uses first order low pass PSP kernel. A pretrained checkpoint locates in ./checkpoint/pretrained_snn_mlp_2. Configuration file is snn_mlp_2.yaml. 
```
  python snn_mlp_2.py --train
```

Following command loads the pretrained model and test it.  
```
  python snn_mlp_2.py --test
```

- associative_memory.py  
A multi-layer fully connected SNN which reconstructs input patterns at output layer. A pretrained model locates in ./associative_memory_checkpoint/pretrained_associative_memory. 
associative_memory.ipynb is the notebook version to inspect input and output.  

## File Organization
- snn_lib: Spiking neural network layers, data loaders,utility functions etc.
- checkpoint: pretrained models.
- dataset: data used for associative memory experiments.
- associative_memory_checkpoint: pre-trained model of associative_memory.py.

## Dependencies
```
torch==1.2.0
numpy==1.19.0
omegaconf==1.4.1
```

## Citation
If this is helpful for you, please cite the following paper:
```
@inproceedings{fang2020exploiting,
	title="Exploiting Neuron and Synapse Filter Dynamics in Spatial Temporal Learning of Deep Spiking Neural Network",
	author="Haowen {Fang} and Amar {Shrestha} and Ziyi {Zhao} and Qinru {Qiu}",
	booktitle="IJCAI 2020: International Joint Conference on Artificial Intelligence",
	notes="Sourced from Microsoft Academic - https://academic.microsoft.com/paper/3034923703",
	year="2020"
}
```
