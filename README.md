# Hackathon Digitaltag 2022
 
Der Hackathon nutzt Daten der Kaggle challange https://www.kaggle.com/kaustubhb999/tomatoleaf

Den Code können Sie über diesen Link in Google Colab öffnen: https://colab.research.google.com/github/Criscraft/HackathonDigitaltag2022/blob/main/TrainingAndValidation.ipynb



## Making the code work offline:

If you prefer to implement this code in a local device (CPU), please follow the below steps: 

- Clone this repository locally, either by downloading the files in zipped format directly from Github or using git library installed in your device:

```bash
git clone https://github.com/Criscraft/HackathonDigitaltag2022.git
```
- Select the folder where you have downloaded/cloned the Git

```bash
cd HackathonDigitaltag2022
```

- We recommend you to use Anaconda for the implementation, for more on installation on Anaconda see: [Ananconda Installation](https://docs.anaconda.com/anaconda/install/) 

- Create a new conda enviornment 

```bash
conda create --name myenv python=3.8 pip
```

- activate the enviorment

```bash
conda activate myenv
```

If you prefer, you can give any desired name for the created enviorment by replacing `myenv` with your desired name

- Install the requirements from the `requirements_cpu.txt` file

```bash
pip install -r requirements_cpu.txt
```

- Once installation is complete, open Jupyter Notebook to make the notebook work

```bash
jupyter notebook
```

- Open jupyter notebook on your favourate web-browser (if not automatically opened)

Open local host [http://localhost:8888](http://localhost:8888)

And select the file `TrainingAndValidation.ipynb` 


