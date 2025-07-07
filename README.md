# imdbFinetuningLLM

### Table of contents
* [Introduction](#star2-introduction)
* [Installation](#wrench-installation)
* [How to run](#zap-how-to-run) 
* [Contact](#raising_hand-questions)

## :star2: Introduction

* <p align="justify">Fine-tuned DistilGPT-2 model on IMDb movie reviews using Hugging Face Transformers.</p>
* <p align="justify">Implemented data preprocessing pipeline and tokenization for text generation tasks.</p>
* <p align="justify">Achieved conditional text generation capabilities from movie-related prompts.</p>

![demo104](/images/demo104.PNG)

Figure: *prompt "The cinematography in this film was visually stunning, with each frame carefully composed to" and max_length 104*

![demo104](/images/demo56.PNG)

Figure: *prompt "Wow, such" and max_length 56*

## :wrench: Installation

<p align="justify">Step-by-step instructions to get you running imdbFinetuningLLM:</p>

### 1) Clone this repository to your local machine:

```bash
git clone https://github.com/hoangnguyen2003/imdbFinetuningLLM.git
```

A folder called `imdbFinetuningLLM` should appear.

### 2) Install the required packages:

Make sure that you have Anaconda installed. If not - follow this [miniconda installation](https://www.anaconda.com/docs/getting-started/miniconda/install).

<p align="justify">You can re-create our conda enviroment from `environment.yml` file:</p>

```bash
cd imdbFinetuningLLM
conda env create --file environment.yml
```

<p align="justify">Your conda should start downloading and extracting packages.</p>

### 3) Activate the environment:

Your environment should be called `imdbFinetuningLLM`, and you can activate it now to run the scripts:

```bash
conda activate imdbFinetuningLLM
```

## :zap: How to run 
<p align="justify">To train imdbFinetuningLLM:</p>

```bash
python main.py --mode train
```

To use my fine-tuned model, download [fine_tuned_model](https://drive.google.com/drive/folders/1DX9Dac8TVnqFVXmjgL1PJYgdP6iA6lA4?usp=sharing), extract it if necessary, and place the `fine_tuned_model` folder inside the `results` directory (create the folder in `imdbFinetuningLLM` if it doesn't exist).

You can generate text directly from the command line:

```bash
python main.py --mode generate --prompt "Wow, such" --max_length 56
```

## :raising_hand: Questions
If you have any questions about the code, please contact Hoang Van Nguyen (hoangvnguyen2003@gmail.com) or open an issue.