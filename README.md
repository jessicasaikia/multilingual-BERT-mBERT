# Multilingual BERT (mBERT)
This repository implements a Multilingual BERT (mBERT) model for performing Parts-of-Speech (POS) Tagging on Assamese-English code-mixed texts.

## Introduction to Parts-of-Speech (PoS) Tagging
PoS tagging is the process of identifying and labeling grammatical roles of words in texts, supporting applications like machine translation and sentiment analysis. While different languages may have their own PoS tags, I have used my own custom PoS tags for this model. The Table below defines the custom PoS tags used in this model-

![Table](https://github.com/jessicasaikia/hidden-markov-model-HMM/blob/main/Custom%20PoS%20tags%20Table.png)

## About Multilingual BERT (mBERT)
It is a pre-trained language model by Google, based on the BERT (Bidirectional Encoder Representations from Transformers) architecture, designed for multilingual tasks. It is trained on text from 104 languages using a masked language model (MLM) objective, where some words in the input are randomly masked, and the model learns to predict them based on context. This training allows mBERT to capture cross-lingual patterns and relationships, making it highly effective for tasks involving multilingual or code-mixed text without the need for explicit translation.

**Algorithm**:
1.	The model imports the required libraries and loads the dataset.
2.	The model tokenises the input sentences using the mBERT WordPiece tokeniser
3.	Tokenised sentences are adjusted to a fixed length by adding [PAD] tokens and generate masks to differentiate valid tokens (1) from padded ones (0).
4.	Each token is aligned with its corresponding POS tag. Subwords inherit the same tag as the original word.
5.	The tokens are mapped to numerical IDs using mBERT’s vocabulary.
6.	The token IDs and attention masks are passed into mBERT to compute contextualised embeddings for each token. These embeddings capture both the meaning of the token and its context within the sentence.
7.	The embeddings are fed into a dense (fully connected) layer to map each token’s embedding to probabilities for all POS tags.
8.	Softmax is applied to convert the output into probabilities for each POS tag.
9.	For each token, the POS tag with the highest probability is selected.
10.	Numerical predictions are mapped back to their original tag format
11.	The POS tags for all tokens in the sentence are returned.

## Where should you run this code?
I used Google Colab for this Model.
1. Create a new notebook (or file) on Google Colab.
2. Paste the code.
3. Upload your CSV dataset file to Google Colab.
4. Please make sure that you update the "path for the CSV" part of the code based on your CSV file name and file path.
5. Run the code.
6. The output will be displayed and saved as a different CSV file.

You can also VScode or any other platform (this code is just a python code)
1. In this case, you will have to make sure you have the necessary libraries installed and datasets loaded correctly.
2. Run the program for the output.
   
## Additional Notes from me
If you need any help or questions, feel free to reach out to me in the comments or via my socials. My socials are:
- Discord: jessicasaikia
- Instagram: jessicasaikiaa
- LinkedIn: jessicasaikia (www.linkedin.com/in/jessicasaikia-787a771b2)

Additionally, you can find the custom dictionaries that I have used in this project and the dataset in their respective repositories on my profile. Have fun coding and good luck! :D
