# Overview

This is the official code of LBoost in paper "Enhancing and Cleaning Labels for Graph Neural Networks"

* The **full version** of the paper can be accessed at this file: `paper_full_version.pdf`
* The **model checkpoints and dataset** are available at this link: https://drive.google.com/drive/folders/1-a4wtZ7w7a_HTFYIYOmICS76acRSGuvt?usp=drive_link.
* Pipelines: First, utilize the LLMAug module to generate key features and initial pseudo-labels for each node. Then, employ the Sentence-BERT to convert each keyword into an initial Embedding that can be accepted by the GNN. Subsequently, employ the LBRMiner module to mine rules. The rules mined will be used to determine the final pseudo-labels for unlabeled nodes in the graph. Finally, retrain the GNN using the graph with enriched feature and pseudo-labels.

# LLMAug

The `LLMAug` folder contains code for extracting key features from text using large language models.

## Software requirements

```shell
python >= 3.10
torch == 2.3.0
transformers == 4.41.2
datasets >= 2.16.0
accelerate >= 0.30.1
peft >= 0.11.1
trl >= 0.8.6
vllm == 0.4.3
CUDA >= 11.6
flash-attn >= 2.3.0
torch_geometric == 2.5.3 # this is for GNN
numpy >= 1.24.2
scikit-learn >= 1.2.2
scipy >= 1.10.1
torch >= 1.13.1
tqdm >= 4.65.0
```

## Base model and hardware requirements

The recommended and default base model is Mistral 7B; however, models from the llama, gemma and ChatGLM series, and others are also supported. If you wish to switch the base model, please modify the template as follows:

- **Mistral Series**
  - download link: https://huggingface.co/mistralai
  - Template: `mistral`

- **Llama Series**
  - download link: https://huggingface.co/meta-llama
  - Template: `llama2/llama3`

- **Gemma Series**
  - download link: https://huggingface.co/google
  - Template: `gemma`

- **GLM  Series**
  - download link: https://huggingface.co/THUDM
  - Template: `glm3/glm4`


For fine-tuning large language models ranging from 7B to 9B parameters, a minimum of one NVIDIA 3090 GPU is required, with a recommended GPU Memory of at least 24GB per card.

For fine-tuning models larger than 13B parameters, at least one V100 GPU is necessary, with a recommended GPU Memory of at least 32GB per card.

## Run

### fine tune

```shell
python main.py train train.yaml
```

```yaml
# train.yaml
### model
model_name_or_path: your_model_path
quantization_bit: 4 # quantization bits, reduce model size and speed up inference, comment out to use LoRA fine-tuning.

### method
neftune_noise_alpha: 5 # Noise rate for NEFTune, adds random noise to embeddings during training to improve fine-tuning performance.
stage: sft # Supervised fine-tuning stage
do_train: true # Flag to indicate training mode
finetuning_type: qlora # Fine-tuning method, can be qlora (quantized LoRA) or other types
lora_target: all # Apply LoRA fine-tuning to all model layers

### dataset
dataset: your_data_set_name
template: mistral # Template for dataset processing, depending on the base model
cutoff_len: 1024 # Maximum sequence length for input data
max_samples: 20 # Maximum number of samples to use from the dataset
overwrite_cache: true # Overwrite cached dataset files if they exist
preprocessing_num_workers: 16 # Number of worker threads for data preprocessing

### output
output_dir: checkpoint_output_path
logging_steps: 10 # Number of steps between logging metrics
save_steps: 500 # Number of steps between saving checkpoints
plot_loss: true # Flag to plot loss during training
overwrite_output_dir: true # Overwrite the contents of the output directory if it exists

### train
per_device_train_batch_size: 1 # Batch size per device during training
gradient_accumulation_steps: 8 # Number of steps to accumulate gradients before updating
learning_rate: 1.0e-4 # Learning rate for the optimizer
num_train_epochs: 3.0 # Total number of training epochs
lr_scheduler_type: cosine # Type of learning rate scheduler (cosine annealing)
warmup_ratio: 0.1 # Proportion of training steps to perform learning rate warmup
fp16: true # Use 16-bit (half precision) floating point arithmetic for training
ddp_timeout: 180000000 # Timeout for distributed data parallel (DDP) training in seconds

### eval
val_size: 0.1 # Proportion of the dataset to use for validation
per_device_eval_batch_size: 1 # Batch size per device during evaluation
eval_strategy: steps # Evaluation strategy to use (evaluate every few steps)
eval_steps: 500 # Number of steps between evaluations
```

The data format needs to be in JSON format, and the dataset must be registered within dataset_info.json. 

**Examples for maple**:

```json
{
    {
        "instruction": "You are an expert in feature engineering in machine learning. Now I will provide you with the information of a paper entity in Computer Science domain. You are expected to automatically conduct a rough categorization for this paper entity based on the given rich text information, and give me the most critical semantic keywords in the information that support you to do this categorization.\n\nThe category should be selected from the following domains:['machine vision', 'knowledge discovery', 'digital media', 'procedure', 'artificial intelligence', 'computational linguistics', 'computational theory', 'modeling', 'cybersecurity', 'feature recognition', 'coding language', 'data storage', 'integrated system', 'software development', 'telecom', 'cloud computing', 'operating system', 'human computer interaction', 'voice recognition', 'networking', 'real-time processing', 'hardware', 'world wide web', 'machine learning', 'parallel processing', 'computer graphics', 'computer engineering', 'data analytics', 'information retrieval', 'computational science', 'knowledge management', 'librarianship', 'computer architecture', 'online privacy']; \n\nThe keywords should have three words.\n\nYour output should be in JSON format. \n\nHere is the entity information I give you:\n\n{'title': 'cross language access to recorded speech in the malach project', 'abstract': 'the malach project seeks to help users find information in a vast multilingual collections of untranscribed oral history interviews this paper introduces the goals of the project and focuses on supporting access by users who are unfamiliar with the interview language it begins with a review of the state of the art in cross language speech retrieval approaches that will be investigated in the project are then described czech was selected as the first non english language to be supported so results of an initial experiment with czech english cross language retrieval are reported', 'venue': 'Lecture Notes in Computer Science'}",
        "input": "",
        "output": "{'category': 'computational linguistics', 'keywords': ['cross language access', 'multilingual oral history', 'czech english retrieval']}"
    },
	...
}
```

**Example for amazon**:

```json
{
	{
        "instruction": "You are an expert in feature engineering in machine learning. Now I will provide you with all the information of an Amazon Clothing Product Entity. You are expected to automatically conduct a rough categorization for this product entity based on the given rich text information, and give me the most critical semantic keywords in the information that support you to do this categorization.\n\nThe category should be selected from the following domains:[\"Men's Fashion\", \"Women's Fashion\", 'Amazon Fashion', 'Halloween Costumes', \"Men's Work Uniforms and Safety Gear\", \"Women's Shoes\", 'Sports and Outdoors', \"Men's Wristwatches\", \"Men's Shoes\", \"Women's Clothing\", \"Women's Lingerie\", \"Men's Clothing\", 'Fashion Jewelry', 'Shoe Care and Accessories']; \n\nThe keywords should have three words.\n\nYour output should be in JSON format. \n\nHere is the entity information I give you:\n\n{'title': \"Rubie's Costume Star Wars Darth Vader Deluxe Adult Costume\", 'label': 'review', 'overall': 4.0, 'vote': '7', 'verified': True, 'reviewTime': '11 12, 2010', 'reviewerID': 'A21THK1P9G55UL', 'asin': 'B000C9X5EM', 'style': {'Size:': ' X-Large', 'Color:': ' Black'}, 'reviewerName': 'Thinkpadius', 'reviewText': \"Overall the suit was good and the cape really looks awesome with even a small amount of wind.  I didn't use the helmet that came with it (it was a mask with no back!) and instead bought a full helmet (worth it.)\\n\\nI did notice that the front flap flipped up a lot, so I tied it down with a clothes pin (its all foam anyway.)\\n\\nOne thing thing prevents me giving it five stars: I didn't get a single woman while wearing this!  I mean, its not like I didn't try. But seriously, Darth Vader has no sex appeal and you should keep that in mind the next time you go to a party dressed like him.\", 'summary': 'Not bad, but hard to get women', 'unixReviewTime': 1289520000}",
        "input": "",
        "output": "{'category': 'Halloween Costumes', 'keywords': ['darth vader deluxe', 'vader deluxe adult', 'costume star wars']}"
    },
	...
}
```

Please note that in order to adapt the LLM to downstream tasks, fine-tuning is a necessary process. This process requires the manual annotation of approximately 20 to 50 training samples, which should cover all categories. Each sample must explicitly indicate the category to which the text belongs and several keywords (this project defaults to using 3 keywords). When manually selecting keywords from text, it is recommended to use a pre-trained GNN as a self-supervised model to ensure the effectiveness of the keyword selection and to avoid introducing redundant information (see **sec5** and **Figure 4(a) in sec7** of `paper_full_version.pdf` for details). 

For the convenience of other developers, this project has already completed this manual annotation process in advance. The annotated samples can be found in the `dataset_name_feature_train.json` file located in the LLM folder of each dataset. Additionally, the checkpoints for LLM fine-tuning have also been uploaded to Google Drive. You can directly merge these LoRA checkpoints to generate a fine-tuned LLM.

### Merge lora checkpoint to get fine-tuned LLMs

```shell
python main.py export merge.yaml
```

```yaml
# merge.yaml
### model
model_name_or_path: your_model_path
adapter_name_or_path: checkpoint_output_path
template: mistral
finetuning_type: qlora # lora

### export
export_dir: merged_model_path
export_size: 2 # model size(GB) of per fragment
export_device: cpu
export_legacy_format: false

```

### Inference

```shell
python -m infer.py model_path dataset_path dataset_name gpu_nums
```

Once the inference of the large language model is completed, you will receive a JSON file formatted as follows, where "category" can serve as a pseudo-label generated by the LLM, and "keywords" can be used for sentence-BERT (such as [SimCSE](https://github.com/princeton-nlp/SimCSE)) to generate the initial embeddings that the GNN initially accepts.

For the convenience of other developers, this project has already pre-completed the inference process and the feature generation process of the LLM. In the `dataset_name_filled.json` file under the LLM folder of each dataset, there are pseudo-labels for rough classification and keywords for feature generation, which have been inferred by the fine-tuned LLM. In the GNN folder of each dataset, the `feature_{keywords_num}.pth` contains the initial embeddings using SimCSE.

# LBRMiner

The `LBRMiner` folder contains code for run LBRMiner.

## Installing dependencies on Ubuntu

GCC version: 7.4.0 or above, support of c++17 standard required.

Install mpi:

```shell
sudo apt-get install openmpi-bin openmpi-doc libopenmpi-dev
```

Install glog:

```shell
sudo apt-get install libgoogle-glog-dev
```

Install gflags:

```shell
sudo apt-get install libgflags-dev
```

Install yaml:

```shell
sudo apt-get install libyaml-cpp-dev
```

The recommended hardware configuration is a single machine with no less than 128GB of memory (refer to  **sec 6** of the `paper_full_version.pdf` for details).

## Compile

```shell
mkdir build && cd ./build
cmake ../
make all -j
```

## Run

We use libgrape-lite for multi-process parallelism and openmp for multi-thread parallelism.

For LBR discovery, to run with single machine, occupying all threads:

```shell
./build/gar_discover --yaml_file ${yaml_file_name}
```

To run with single machine, occupying a specified number of threads:

```shell
mpirun -n 1 -map-by slot:pe=core_num ./build/gar_discover --yaml_file ${yaml_file_name}
```

To run with multiple machines:

```shell
mpirun -N xxx -n yyy -c zzz ./build/gar_discover --yaml_file ${yaml_file_name}
```

For rule match, to run with single machine, occupying all threads:

```shell
./build/rule_match --yaml_file ${yaml_file_name}
```

The others are same as discovery.

The main loop of LBR discovery can be found in folder LBoostMiner/src/apps/rule_discover/, and the main loop of LBR match, error detection can be found in folder LBoostMiner/src/apps/rule_match/.

The BiSimulation folder contains code for computing graded bisimilarity, where you can employ any feature (such as SimCSE or GloVe) embedding to determine whether a pair of points, as well as all pairs within a graph, exhibit BiSimulation. Detailed running examples can be viewed within this folder. You may need to send pairs of BiSimulation to LBR candidates, or simply need to mark the point pairs with BiSimulation with a distinct label, and define this label within the parameter of `SpecifiedRhsLiteralSet` to complete the mining of the dual star pattern containing the BiSimulation relationship.

If you want to run LBR discovery algorithm, you may need to fill a yaml file in this format:

```yaml
DataGraphPath: # the path for the data graphs
  - VFile: the vertex file for the first data graph
    EFile: the edge file for the first data graph
    MlLiteralEdgesFile: (optional) the edges that are added by the well-trained ml model for the first data graph
  ...
ExpandRound: number of expand round, i.e. total edges to be added
J: depth of the literal tree for horizontal spawning.
SupportBound: the support bound for the gar to be discovered
OutputGarDir: the directory for the discovered gar to export
TimeLimit: time limit for evaluating the support bound of each gar or graph pattern
TimeLimitPerSupp: time limit for it to complete the match of the entire pattern of gar at each support
ConstantFreqBound: the frequency bound for the constant, only the value appear larger than this frequence would be considered
PatternVertexLimit: the limit of pattern vertex
DiameterLimit: the limit of the diameter of the graph pattern
LiteralTypes: # the literal types to be considered
  - constant_literal
  - variable_literal
  - edge_literal
Restrictions: # the restrictions for the gar
  - variable_literal_only_between_connected_vertexes
  - edge_literal_only_between_2_hop_connected_vertexes
  - literals_connected
  - pattern_without_loop
SpecifiedRhsLiteralSet:
  - Type: variable_literal
    XLabel: label of x
    YLabel: label of y
    XAttrKey: attr of x
    YAttrKey: attr of y
TimeLogFile: the path for the time log file
```

An example yaml file for LBR discovery may like this:

```yaml
DataGraphPath:
  VFile : dataset/business/business_v.csv
  EFile : dataset/business/business_e.csv
ExpandRound: 15
J: 3
LiteralTypes:
  - constant_literal
  - variable_literal
  - edge_literal

SupportBound: 1
ConfidenceBound: 0.4

Rule:
  Type: gcr
  PathNumLimit: 3
  PathLengthLimit: 5
  
SpecifiedRhsLiteralSet:
  - Type: variable_literal
    XLabel: 3
    YLabel: 3
    XAttrKey: year
    YAttrKey: year
  
TimeLogFile:  dataset/lboost_business.log
OutputGarDir: dataset/lboost

TimeLimit: 3000
TimeLimitPerSupp: 0.5
ConstantFreqBound: 0.09
```

If you want to run LBR error detection algorithm, you may need to fill a yaml file in this format:

```yaml
DataGraphPath: 
  VFile : vertex file of the data graph
  EFile : edge file of the data graph
  
PatternPath:
  VFile : vertex file of the pattern
  EFile : edge file of the pattern
  XFile : X (lhs) literal file of the pattern
  YFile : Y (rhs) literal file of the pattern
  PivotId : (optional) specify the pivot vertex id, needs to be contained in the Y literals of the pattern

TimeLogFile: time log file
```

An example yaml file for LBR error detection may like this:

```yaml
DataGraphPath:
  VFile : dataset/business/business_v.csv
  EFile : dataset/business/business_e.csv

MatchSemantics: homo
RuleType: gcr

RulePath:
  - VFile : dataset/rule/lbr_1_v.csv
    EFile : dataset/rule/lbr_1_e.csv
    XFile : dataset/rule/lbr_1_x.csv
    YFile : dataset/rule/lbr_1_y.csv

  - VFile : dataset/rule/lbr_2_v.csv
    EFile : dataset/rule/lbr_2_e.csv
    XFile : dataset/rule/lbr_2_x.csv
    YFile : dataset/rule/lbr_2_y.csv

LogFile: business_log
```

For the convenience of other developers, we have uploaded some of the mined rules along with the dataset and model checkpoints to Google Drive. You can directly utilize these mined rules to supplement pseudo-labels.

# GNNs

The `GNNs` folder contains code for run Graph Neural Networks.

Before generating pseudo labels and noise, we need to pre-train an initial GNN. To train or test a single GNN, run `train.py` and `test.py`To batch train all GNNs, run `batch_train.py` and `batch_test.py`

You can directly modify the training parameters for GNNs in `args.py`.

```python
def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', default=1e-4, type=float, help="learning rate")
    parser.add_argument('-hidden_dim', default=256, type=int, help="embedding dimension")
    parser.add_argument('-weight_decay', default=1e-6, type=float, help="l2 regularization parameter")
    parser.add_argument('-epoch', default=500, type=int, help="training epoch")
    parser.add_argument('-noise_ratio', default=20, type=int, help="noise ratio")
    parser.add_argument('-model_type', default="GCN", type=str, help="model type")
    parser.add_argument('-dataset', default="Office", type=str, help="dataset name")
    parser.add_argument('-cleaned', default=True, type=bool, help="if data is cleaned")
    parser.add_argument('-feature', default=1, type=int, help="feature num")
    parser.add_argument('-soft', default=False, type=bool, help="Whether to use soft label")
    args = parser.parse_args()
    return args

```

This function configures the parameters for a Graph Neural Network (GNN) training process. Below are the parameters it accepts: - `-lr` (float, default=1e-4): The learning rate, which controls the step size during the optimization process.  `-hidden_dim` (int, default=256): The dimension of the hidden layers within the GNN, affecting the model's capacity.  `-weight_decay` (float, default=1e-6): The L2 regularization parameter, used to prevent overfitting by penalizing large weights.  `-epoch` (int, default=500): The number of training epochs, which is the number of times the entire dataset is passed forward and backward through the network. `-noise_ratio` (int, default=20): The ratio of noise to be added to graph.  `-model_type` (str, default="GCN"): The type of GNN model to be trained, such as GCN, GAT, GraphSAGE. `-dataset` (str, default="Office"): The name of the dataset to be used for training and evaluation.  `-cleaned` (bool, default=True): A flag indicating whether the dataset has been pre-cleaned for training.  `-feature` (int, default=1): The number of features for each data point in the dataset.  `-soft` (bool, default=False): A flag to determine if soft labels should be used during training for a probabilistic approach. 

## Inject Noise by MGAttack

This project employs noise injection via **MGAttack** (refer to **sec6** of `paper_full_version.pdf` for details). The noise injection ratio varies from 5% to 20%. If you wish to freely choose the noise injection ratio, modify the array on line 23 of the code to the desired noise injection ratio:

```python
error_rates = [5, 10, 15, 20]
```

To perform noise injection, run `python -m mgattack.py` directly.

## Soft labeling

If multi-label nodes occur during the label imputation chase process, we use `generate_softlabels.py` to generate soft labels for these nodes, and you must set the `-soft` flag to `True` in the arguments.

You need to enter the conflicting nodes and the conflicting categories into the following data structure in line 29 of the code:

```python
# Fill this list with node that had been assigned multi-labels in LBR Chase procedure
multi_labels_nodes = {
    0: [7, 21],
    124: [5, 9],
    345: [4, 8, 11, 16]
}
```

For the convenience of other developers, we have pre-run the aforementioned programs and uploaded the files with injected noise, the files cleaned of noise using LBR, model checkpoints, and soft label files to Google Drive. You can simply place them respectively under the `GNNs/data` and `GNNs/models` directories and run them with just one click.

# Acknowledgements

This project has benefited from the following open-source projects, to which we extend our gratitude:

- **torch_geometric**: https://github.com/pyg-team/pytorch_geometric
- **Mistral**: https://huggingface.co/mistralai
- **SimCSE**: https://github.com/princeton-nlp/SimCSE
- **vllm**: https://github.com/vllm-project/vllm
- **LLaMA-Factory**: https://github.com/hiyouga/LLaMA-Factory
- **GUNDAM**: https://github.com/MinovskySociety/GUNDAM

We acknowledge these contributions, which have significantly facilitated the development of our project.