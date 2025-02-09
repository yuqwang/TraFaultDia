# Replication package 

This repo contains the implementation details for the paper "Cross-System Categorization of Abnormal Traces in
Microservice-Based Systems via Meta-Learning"
## File structure and description
 - OnlineBoutiqueFaultList: a list of base fault categories and novel fault categories in our study
 - TrainticketFaultList: a list of base fault categories and novel fault categories in our study
 - models:
   - MAML.py: the implementation of the MAML algorithm. The MAML class manages the meta-training and meta-testing phases. It trains the base model in Learner.py. We considered this repo MAML-Pytorch in our implementation.
   - Learner.py: defines the base model that do abnormal trace classification 
   - AttenAE.py: the implementation of AttenAE
   - DatasetFusion.py: load unlabled traces into the Dataset
   - AttenAE_train.py: train AttenAE using unlabled traces for each MSS
   - DatasetMix.py: Dataset for cross-system contexts
   - DatasetTT.py: Dataset only for Trainticket-Trainticket
   - DatasetOB.py: Dataset only for OnlineBoutique-OnlineBoutique
   - preprocessing: 
   - TrainMix.py: train TEMAML in cross-system contexts using Ray
   - TrainOB.py: train TEMAML for Trainticket-Trainticket contexts
   - TrainTT.py: train TEMAML for OnlineBoutique-OnlineBoutique contexts using Ray
  - imple_details: hyperparameter settings for training AttenAE and TEMAML
  - requirements.txt: lists the dependencies (packages and their versions) required to run the project.


### Datasets
 - This project uses open datasets: Nezha (https://github.com/IntelligentDDS/Nezha) and DeepTraLog (https://fudanselab.github.io/DeepTraLog/).  We do not hold the right to publish these datasets here. Please refer to the original sources for downloading the datasets.


