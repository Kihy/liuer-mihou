# Liuer Mihou
## A Practical Adversarial Generation Framework

***Note: The exact Kitsune training parameters and surrogate parameters may be different from the paper.***

### Installation
- The following commands assume that you are using mamba.
If you don't have one, install mamba (or conda). 
See [conda-forge/miniforge](https://github.com/conda-forge/miniforge#install).

```commandline
mamba create -n lm
mamba activate lm
mamba install python=3.8 pandas numpy matplotlib scikit-learn openpyxl tqdm ipython jupyter cython

pip install tensorflow pyswarms rrcf minisom
```

### Folder structure
- /code contains code for pso framework
The following folders are omitted due to size constraints
- /dataset contains pcap files used for experiments, replace with your own dataset


### Code
- /after_image contains original and modified implementation of Kitsune's feature extractor, AfterImage. Files with CamelCase are original implementation and files with underscore is modified implementation.
- /evaluations contains scripts to to evaluate the attacks and models
- /KitNET contains implementation of Kitsune's anomaly detector
- /topology contains our PSO-DE search algorithm
- after_image_test.py contains several tests to test functionalities of modified after_image.
- kitsune.py contains code to train, save and evaluate Kitsune
- parse_with_kitsune.py contains code that parses pcap files with AfterImage, parsed values are features saved in CSV files
- pso.py is the main LiuerMihou framework
- pso_framework.py defines hyperparameters that runs the LiuerMihou framework
- surrogate_model.py train, save and evaluate the surrogate autoencoder model.
- run_experiments.py contains code to run entire experiments, should be the main script to customize.
- vae.py contains code for variational autoencoder.

### Running this code
1. gather some normal traffic of your network, Google_Home_Mini sample is provided in dataset folder.
2. gather some attack traffic with your attack, port scan, OS/service detection, and HTTP flooding samples are provided in dataset folder.
3. parse pcap files to csv files with parse_with_kitsune.py the csv file is in the same directory as the original
4. train kitsune with train_normal() in kitsune.py and train surrogate with train() in surrogate_model.py. the trained models are in /models
5. eval kitsune and surrogate model with eval() and eval_surrogate() with normal traffic to find the threshold value.
6. (optional) evaluate the attack traffic on kitsune and surrogate model to see its anomaly pattern
7. (optional) run /evaluations/similarity.py to see similarities between attacks. The script has to be run under /code directory
8. run pso_framework.py to generate adversarial samples

The above steps are also outlined in run_experiments.py

### Results structure
By default, all results are saved in /experiment/traffic_shaping/{attack name}. Under the folder there are several folders
- /png contains plots of anomaly Scores
- /meta contains meta files
- /logs contains logs and reports of each run
- /csv contains extracted features for adversarial samples
- /craft contains pcap files without init_pcap packet
- /anim contains animations of position history in search algorithm
- /adv contains adversarial packets with normal traffic
