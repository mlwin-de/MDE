# MDE : Multiple Distance Embeddings for Link Prediction in Knowledge Graphs

This is the PyTorch implementation of MDE. The implementation is tailored for cpu servers and performs distributed testing using 8 CPU cores. A GPU version that includes Self-Adversarial Negative Sampling is implemented in [here](https://github.com/mlwin-de/MDE_adv).


** Training ** :
 To train the model from the command line :

 python MDE_Model.py -t task -d dataset_name

Where the task is “train” here and “dataset_name” can be one of WN18, WN18Rr, FB15K, and FB15K237



Or for MDE_NN:

 python MDE_NN_Model.py -t task -d dataset_name


For example: 
python MDE_Model.py -t train -d WN18RR 

During the training, a test will be executed after every 50 iterations.

**Citation**
If you use the codes, please cite the following paper:
```
@inproceedings{sadeghi2019mde,
  title={MDE: Multiple distance embeddings for link prediction in knowledge graphs},
  author={Sadeghi, Afshin and Graux, Damien and Hamed, Shariat Yazdi and Lehmann, Jens},
  conference={ECAI},
  year={2020}
}
```
