# LLM-Driven Baselines for Medical Image Segmentation

This project utilizes Large Language Models (LLMs) to generate code for medical image segmentation tasks using U-Net architectures. We compare the performance of various LLMs in generating code for U-Net-based segmentation models and track the interactions with the LLMs and results on multiple datasets. The models are evaluated on their ability to generate functional code out-of-the-box, accuracy in segmentation, and error frequency during development.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation and Usage](#installation)
- [Models and LLMs](#models-and-llms)
- [Datasets](#datasets)
- [Results](#results)
- [License](#license)


---

## Project Overview
This repository explores how Large Language Models (LLMs) perform when tasked with generating code for medical image segmentation. We used eight state-of-the-art LLMs, including GPT-4, 4o and o1 Preview, Claude 3.5 Sonnet, Gemini 1.5 Pro, Github Copilot, Bing Microsoft Copilot (closed source) and LLAMA 3.1 405B (open source), to compare performance across different models for U-net based semantic segmentation on several medical image datasets, focusing on the ease of generation, minimum required modifications and error frequency, and compare the performance across the different LLM generated model outputs.
The models were prompted with the same engineered prompt, and asked to generate a dataloader, training, model and main script. Errors were fed back to the model and suggested fixes were used to debug the codes untill it ran error-free.

We use the **Dice coefficient** as the key evaluation metric for segmentation quality and track error rates and types, and interactions required with each LLM to make the code run through. The final models are evaluated on three datasets to measure their performance on real-world medical image segmentation tasks.

For benchmarking model performance, we used [nnUnet v2](https://github.com/MIC-DKFZ/nnUNet)
 as a well-established baseline in biomedical image segmentation. nnU-Net provides an automated, self-configuring framework that enables fair and reliable comparisons across various segmentation tasks [(Isensee et al., 2021)]

---

## Features
- **Multiple LLM Comparisons**: Assess how different LLMs perform in generating code for a U-Net-based segmentation model.
- **Engineered Prompt**: Detailed prompt to generate scripts for dataset, train, model and main scripts is provided in Prompt_final.txt file, and can be modified and tailored based on needs or to test any additional LLM.
- **Medical Datasets**: Utilized real medical image datasets for training and evaluation.
- **Dice Score Evaluation**: Compare Dice scores for each model on different datasets.
- **Error Tracking**: Record number of errors, error types and debugging for each LLM.
- **Generated U-Net Architectures**: Each U-net model is trained using the out-of-box architecture generated by the different LLMs.

---

## Installation and Usage

### Prerequisites
Ensure you have the following installed:
- Python 3.6 or higher
- PyTorch
- CUDA (optional for GPU support)

### Steps to Install

1. Clone the repository:
   ```bash
   git clone https://github.com/ankilab/LLM_based_Segmentation.git
   cd LLM_based_Segmentation
2. Installing required packages:
   pip install -r requirements.txt

### Usage
#### Running the Models:
You can train and evaluate the models using the scripts and main.py provided for each LLM-generated architecture, in it's respective folder.
The prompt text in Prompt_final.txt file can directly be used or tailored according to needs, to generate the dataset, train, model and main scripts using any other LLM. The dataloader in Dataset.py can also be modified based on dataset requirements.
For comparison of performances across models, the scripts in Models Comparison can be used to visualize and compare the validation and test dice scores, as well as the train and validation losses across models, and run inference on a single example image from the dataset for each model.

### Additional Material
- The full chat script with each LLM from prompt input to final error correction was added as a .json file in each LLM's respective folder.
- The prompt engineering process documentation was added as .docx in Models Comparison/Results folder
- The tables for initial comparison of all features for dataloader, models, train and main scripts across models were added as excel sheets in Models Comparison/Results folder

## Models and LLMs
This project uses the following LLMs to generate U-Net architectures:

GPT-4 (Closed-source)
GPT-4o (Open-source)
Claude 3.5 Sonnet (Open-source)
LLAMA 3.1 405B (Open-source)
Gemini 1.5 Pro (Open-source)
Bing Microsoft Copilot (Closed-source)
Copilot (Closed-source)
GPT-o1 Preview (Open-source)

Key differences between these models include the number of encoder/decoder stages, convolutional block design, use of batch normalization, skip connections, bottleneck size, and final activation functions, as well as choice of hyper parameters, such as number of epochs, batch size and image resizing. These variations contribute to differences in segmentation performance and ease of model generation.

| **Model**                  | **Company**         | **Model Version**      | **Open Source** | **Character Limit** | **Token Limit** |
|----------------------------|---------------------|------------------------|-----------------|---------------------|-----------------|
| **GPT-4**                  | OpenAI              | GPT-4                  | No               | 25,000              | 8,192           |
| **GPT-4o**                 | OpenAI              | GPT-4o                 | No               | 50,000              | 32,768          |
| **GPT o1 Preview**         | OpenAI              | GPT o1 Preview         | No               | 50,000              | 128,000         |
| **Claude 3.5 Sonnet**      | Anthropic           | Claude 3.5 Sonnet      | No               | 90,000              | 100,000         |
| **LLAMA 3.1 405B**         | Meta AI             | LLAMA 3.1              | Yes              | 30,000              | 32,768          |
| **Gemini 1.5 Pro**         | Google DeepMind     | Gemini 1.5 Pro         | No               | 25,000              | 8,192           |
| **GitHub Copilot**         | GitHub (Microsoft)  | Codex-based Copilot    | No               | 16,000              | 4,000           |
| **Bing Microsoft Copilot** | Microsoft           | GPT-4-powered          | No               | 32,768              | 8,192           |

**Table**: Comparison of the selected LLMs by input character limit, token limit, and open-source status.

## Datasets
We evaluated the LLM-generated models on three standard medical image segmentation datasets: the Benchmark for Automatic Glottis Segmentation ([BAGLS](https://doi.org/10.1038/s41597-020-0526-3)), an internal Bolus Swallowing Dataset (without a pathological condition), and a [Brain Tumor Dataset](https://doi.org/10.1016/j.eswa.2023.122347) . The BAGLS dataset consists of endoscopic video frames from laryngoscopy exams for glottis segmentation. The Bolus Swallowing Dataset provides videofluoroscopic swallowing studies to assess swallowing disorders. The Brain Tumors dataset includes single-slice grayscale MR images with segmentation masks for different types of brain tumors, including Glioma, Meningioma, and Pituitary tumors, which for this study, we only used the Meningioma tumor images and their corresponding masks.

### Preprocessing
All datasets were preprocessed to fit the input requirements of the LLM-generated models. 5002 total images were selected for the BAGLS and Swallowing dataset and 999 total images for the Brain Meningioma tumor dataset respectively. Images were resized and normalized to a range between 0 and 1 in the LLM-generated code. Each dataset was split into training (80%), validation (10%), and testing (10%) sets with different methods by each LLM.
The batch sizes, learning rate and number of epochs for training were also determined by the LLM.



## Results
### Key Metrics and Architecture differences
1. Model Architecture and Hyper parameter differences:

| **Feature**                      | **GPT-4** | **GPT-4o** | **GPT-o1** | **Claude 3.5** | **Copilot** | **Bing Copilot** | **LLAMA 3.1** | **Gemini 1.5 Pro** |
|-----------------------------------|-----------|------------|------------|----------------|-------------|------------------|---------------|--------------------|
| Batch Size                        | 16        | 8          | 16         | 16             | 16          | 16               | 32            | 8                  |
| Epochs                            | 25        | 25         | 10         | 50             | 25          | 25               | 10            | 20                 |
| Optimizer and Learning Rate       | Adam (lr=0.001) | Adam (lr=1e-4) | Adam (lr=1e-4) | Adam (lr=0.001) | Adam (lr=1e-4) | Adam (lr=1e-4) | Adam (lr=0.001) | Adam (lr=1e-4) |
| Loss Function                     | BCEWithLogitsLoss | BCELoss | BCELoss | BCELoss | BCELoss | BCELoss | BCELoss | BCELoss |
| Image Size                        | 256x256   | 256x256    | 256x256    | 256x256        | 128x128     | 256x256          | 256x256       | 256x256            |
| Number of Encoder Stages          | 4         | 5          | 4          | 4              | 4           | 4                | 3             | 4                  |
| Number of Decoder Stages          | 3         | 4          | 4          | 4              | 4           | 4                | 3             | 4                  |
| Convolutional Block               | Double Conv (Conv2d + ReLU ×2) | Conv2d + BatchNorm2d + ReLU ×2 | Double Conv (Conv2d + BatchNorm + ReLU ×2) | Double Conv (Conv2d + BatchNorm2d + ReLU ×2) | Conv2d + BatchNorm2d + ReLU ×2 | Conv2d + ReLU ×2 | Conv2d + ReLU | Conv2d + BatchNorm2d + ReLU ×2 |
| Bottleneck Layer                  | 512 channels | 1024 channels | 1024 channels | 1024 channels | 512 channels | 512 channels | 256 channels | 1024 channels |
| Final Layer                       | Conv2d(64, 1, 1) | Conv2d(64, 1, 1) | Conv2d(64, 1, 1) | Conv2d(64, 1, 1) | Conv2d(64, 1, 1) | Conv2d(64, 1, 1) | Conv2d(64, 1, 1) | Conv2d(64, 1, 1) |
| Encoder Channels                  | 64, 128, 256, 512 | 64, 128, 256, 512, 1024 | 64, 128, 256, 512, 1024 | 64, 128, 256, 512, 1024 | 64, 128, 256, 512 | 64, 128, 256, 512 | 64, 128, 256 | 64, 128, 256, 512 |
| Decoder Channels                  | 512, 256, 128, 64 | 512, 256, 128, 64 | 512, 256, 128, 64 | 512, 256, 128, 64 | 512, 256, 128, 64 | 512, 256, 128, 64 | 256, 128, 64 | 512, 256, 128, 64 |
| Total Trainable Model Parameters  | 7,696,193 | 31,042,369 | 31,042,369 | 31,042,369     | 6,153,297   | 6,147,659        | 533,953        | 31,042,369         |
| Total Training Time (sec)         | 1474.03   | 949.33     | 780.13     | 5285.17        | 184.16      | 497.04           | 234.27         | 1066.50            |

**Table**: Comparison of Features Across Different LLM-based Models.

#### Model Comparison

1. **Encoder and Decoder Stages**:
    - Most models have a typical 4-stage encoder-decoder setup, except for GPT-4o, which uses 5 encoder stages, and LLAMA 3.1 405B, which employs only 3 encoder-decoder stages. This variation might influence the models' depth and feature extraction capabilities, with more stages generally allowing deeper feature hierarchies.

2. **Convolutional Block Design**:
    - GPT-4, GPT-o1, and Claude 3.5 Sonnet rely on "Double Conv" blocks with two convolution layers followed by ReLU activations. GPT-4o, Copilot, and Gemini 1.5 Pro, incorporate BatchNorm2d after each convolution for added normalization, which helps in stabilizing training.

3. **Bottleneck Layer**:
    - The bottleneck layer varies significantly across models, with models like GPT-4o, GPT-o1, Claude, and Gemini having larger bottlenecks (1024 channels), potentially allowing more complex representations. Meanwhile, LLAMA 3.1 405B has a relatively smaller bottleneck with 256 channels, likely affecting its capacity for encoding detailed spatial information.

4. **Final Layer and Output Channels**:
    - All models conclude with a similar final layer configuration of `Conv2d(64, 1, 1)`, suggesting a standardized approach to mapping the output to a single-channel segmentation mask. However, variations in encoder and decoder channel sizes (e.g., LLAMA 3.1 with fewer channels in the decoder) could impact segmentation output consistency.

5. **Parameter Counts**:
    - The number of trainable parameters differs considerably, with models like GPT-4o, GPT-o1, Claude, and Gemini featuring over 31 million parameters due to deeper layers and larger bottlenecks. In contrast, LLAMA 3.1 has the smallest parameter count (533,953), reflecting a more lightweight architecture.

6. **Training Configuration and Duration**:
    - The training duration also varied greatly, with Claude 3.5 Sonnet taking the longest at over 5,000 seconds, likely due to its higher number of epochs (50). On the other hand, Copilot and LLAMA 3.1 had shorter training times, reflecting simpler architectures and fewer training epochs.

This comparative analysis highlights how architectural and hyperparameter choices across LLM-generated U-Net models impact computational complexity, model depth, and training efficiency. These differences provide insights into balancing model complexity with training duration, essential for selecting models suited to varying computational resources.


### Error Comparison
Errors and interactions with the LLM to fix the errors were tracked and logged. The errors were fed back to the LLM and the suggested fix was applied, until the code was could run through. For LLAMA 3.1 and Gemini 1.5 some additional explanation and input was needed for resolving some errors. GPT-o1 Preview and Claude had 0 errors and ran successfully without modifications, while others such as Gemini and LLAMA required more bug fixes.


<img src="Models Comparison/Results/errors.jpg" alt="Errors comparison across the different models" width="500" height="300">

### Training & Validation Losses

<img src="Models Comparison/Results/all_model_losses_logarithmic_baseline_CE_BAGLS.png" alt="Training and Validation losses across different models" width="600" height="250">

The graphs illustrate the training and validation loss per epoch for each model on the BAGLS dataset as an example, with losses plotted on a logarithmic scale for improved clarity.

#### Training Loss
- Most models, such as **Claude 3.5 Sonnet**, **GPT-o1 Preview**, **GPT-4o**, and **Gemini 1.5 Pro**, exhibit rapid convergence in training loss within the first 10 epochs, indicating efficient learning.
- **Copilot** and **Bing Microsoft Copilot** show a slower and minimal reduction in training loss, which may suggest underfitting or optimization issues.
- The **nnUnet** baseline demonstrates a stable and low training loss across epochs, showcasing its reliability as a baseline model for comparison.

#### Validation Loss
- **Claude 3.5 Sonnet**, **Gemini 1.5 Pro** and **GPT-o1 Preview** achieve the lowest validation loss consistently, suggesting strong generalization performance on the validation set.
- **GPT-4** and **GPT-4o** display higher fluctuations in validation loss, possibly due to instability in the training process.
- **Copilot** and **Bing Microsoft Copilot** showed almost no convergence at very high loss values. **LLAMA 3.1 405B** displayed gradual convergence but remained less converged compared to other models.

In summary, while all models demonstrate some level of convergence, **Claude 3.5 Sonnet** and **GPT-o1 Preview** excel in training stability and validation performance, marking them as robust options among the tested models.


### Validation and Test Dice Scores
Dice scores varied significantly across datasets, with **GPT-4o** and **GPT-o1 preview** and **Gemini 1.5 Pro** models showing better performance overall, and closer to nnU-net baseline dice scores.

1. for BAGLS dataset:

   <img src="Models Comparison/Results/all_model_dice_scores_baseline_BAGLS.png" alt="Validation and Test Dice scores comparison across different models" width="600" height="300">
   

2. for Bolus dataset:
   
   <img src="Models Comparison/Results/all_model_dice_scores_baseline_BOLUS.png" alt="Validation and Test Dice scores comparison across different models" width="600" height="300">
   

3. for Brain Meningioma Tumor dataset:
   
   <img src="Models Comparison/Results/all_model_dice_scores_baseline_BRAIN.png" alt="Validation and Test Dice scores comparison across different models" width="600" height="300">
   

### Performance
The performance differences observed quantitatively with the losses and dice scores comparison, can also be seen in the inference and visualization of the predictions of each model. 

<img src="Models Comparison/Results/inference.jpg" alt="Inferences on example image from each dataset, across different models" width="900" height="350">

**Figure**: Inference prediction masks comparison across LLM generated models along with nnU-net baseline. Each horizontal set shows performance of each model on a sample image from each of the three datasets.


## License

This project is licensed under the Apache License, Version 2.0. You may obtain a copy of the License at:

[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.




