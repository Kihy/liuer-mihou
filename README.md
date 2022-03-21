# Liuer Mihou
## A Practical Adversarial Generation Framework

***Note the exact Kitsune training parameters and surrogate parameters is different from the paper***

### Folder structure
- /code contains code for pso framework
The following folders are omitted due to size constraints
- /experiment contains experiment results
- /ku_dataset contains pcap files used for experiments, replace with your own dataset
- /models contains trained anomaly detection models

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
1. pull docker image from [docker hub](https://hub.docker.com/repository/docker/kihy/deepo-nids). The image depends on [Deepo](https://github.com/ufoym/deepo) so install the dependencies for that first.
2. run commands in commands.sh to start docker if you are on windows, linux should change the directory names and goto /code folder. All code should be run at this level.
3. gather some normal traffic of your network, Google_Home_Mini sample is provided in dataset folder.
4. gather some attack traffic with your attack, Port scan is provided in dataset folder.
5. parse pcap files to csv files with parse_with_kitsune.py the csv file is in the same directory as the original
6. train kitsune with train_normal() in kitsune.py and train surrogate with train() in surrogate_model.py. the trained models are in /models
7. eval kitsune and surrogate model with eval() and eval_surrogate() with normal traffic to find the threshold value.
8. (optional) evaluate the attack traffic on kitsune and surrogate model to see its anomaly pattern
9. (optional) run /evaluations/similarity.py to see similarities between attacks. The script has to be run under /code directory
10. run pso_framework.py to generate adversarial samples

The above steps are also outlined in run_experiments.py

### Results structure
By default all results are saved in /experiment/traffic_shaping/{attack name}. Under the folder there are serveral folders
- /png contains plots of anomaly Scores
- /meta contains meta files
- /logs contains logs and reports of each run
- /csv contains extracted features for adversarial samples
- /craft contains pcap files without init_pcap packet
- /anim contains animations of position history in search algorithm
- /adv contains adversarial packets with normal traffic
