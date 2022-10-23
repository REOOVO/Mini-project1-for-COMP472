# COMP 472 Mini Project 1
https://github.com/REOOVO/Mini-project1-for-COMP472
## Team Members
Haoxuan Lyu 40087583

Tianfei Qi 40079518

## How to run the program

### Use Colab (Recommended)
1. Open the `A1.ipynb` in Colab
2. At lease, upload the `goemotions.json.gz` and `Dataset_Preparation.py` to Colab
3. Run the code in the notebook

Please note that the code may take a long time to run, so please be patient.

Also, there exists compatibility issues due to the version of the `gensim`. If you want to run Jupyter Notebook locally, please install the `gensim` version 3.6.0.
### Use python
1. Install the `gensim` version >= 4.0.0
2. Put the `goemotions.json.gz` in this directory as show in the file tree
2. Run the `Dataset_Preparation.py` to generate the `emotion.jpg` and `sentiment.jpg`
3. Run the `Words_as_Features.py` to generate the part 2 of the mini project 1, the performance of the model will be printed in the console and save at `performance.txt` and `tf_performance.txt`
4. Run the `Embeddings_as_Features.py` to generate 3.1 to 3.7 of the mini project 1. The performance of the model will be printed in the console and save at `embedding.txt`
5. Run the `conceptnet.py` to generate 3.8 which used base_MLP model with `conceptnet-numberbatch-17-06-300`. The performance of the model will be printed in the console and save at `conceptnet.txt`
6. Run the `twitter.py` to generate 3.8 which used base_MLP model with `glove-twitter-200`. The performance of the model will be printed in the console and save at `twitter.txt`

## Files in Project

### File Tree
```
+--A1.ipynb
+--analysis.docx
+--conceptnet.py
+--conceptnet.txt
+--Dataset_Preparation.py
+--embedding.txt
+--Embeddings_as_Features.py
+--goemotions.json.gz
+--graph
|      +--emotion.jpg
|      +--sentiment.jpg
+--performance_full.txt
+--performance_full_MLP.txt
+--plots.pdf
+--README.md
+--tf_performance_full.txt
+--tf_performance_full_MLP.txt
+--twitter.py
+--twitter.txt
+--Words_as_Features.py
```
#### Performance
performance_full.txt: performance of the model without MLP

performance_full_MLP.txt: performance of the MLP model

tf_performance_full.txt: performance of the TfidfTransformer model without MLP

tf_performance_full_MLP.txt: performance of the TfidfTransformer MLP model

embedding.txt: performance of the embedding model which used `word2vec-google-news-300`

conceptnet.txt: performance of the embedding model which used `conceptnet-numberbatch-17-06-300`

twitter.txt: performance of the embedding model which used `twitter-2016-10-21`

#### Plots
plots.pdf: plots of `goemotions.json.gz`
graph/emotion.jpg: plots of `goemotions.json.gz` with emotion labels
graph/sentiment.jpg: plots of `goemotions.json.gz` with sentiment labels

#### Analysis
analysis.docx: analysis of the project

