# Quora-Question-Pairs


1. Business Problem 

1.1 Description 


Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.


> Credits: Kaggle
__ Problem Statement __

Identify which questions asked on Quora are duplicates of questions that have already been asked.
This could be useful to instantly provide answers to questions that have already been answered.
We are tasked with predicting whether a pair of questions are duplicates or not.
1.2 Sources/Useful Links
Source : https://www.kaggle.com/c/quora-question-pairs

____ Useful Links ____
Discussions : https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb/comments
Kaggle Winning Solution and other approaches: https://www.dropbox.com/sh/93968nfnrzh8bp5/AACZdtsApc1QSTQc7X0H3QZ5a?dl=0
Blog 1 : https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning
Blog 2 : https://towardsdatascience.com/identifying-duplicate-questions-on-quora-top-12-on-kaggle-4c1cf93f1c30
1.3 Real world/Business Objectives and Constraints 
The cost of a mis-classification can be very high.
You would want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice.
No strict latency concerns.
Interpretability is partially important.
2. Machine Learning Probelm 
2.1 Data 
2.1.1 Data Overview 
- Data will be in a file Train.csv
- Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate
- Size of Train.csv - 60MB
- Number of rows in Train.csv = 404,290

2.1.2 Example Data point 
"id","qid1","qid2","question1","question2","is_duplicate"
"0","1","2","What is the step by step guide to invest in share market in india?","What is the step by step guide to invest in share market?","0"
"1","3","4","What is the story of Kohinoor (Koh-i-Noor) Diamond?","What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back?","0"
"7","15","16","How can I be a good geologist?","What should I do to be a great geologist?","1"
"11","23","24","How do I read and find my YouTube comments?","How can I see all my Youtube comments?","1"
2.2 Mapping the real world problem to an ML problem 
2.2.1 Type of Machine Leaning Problem 
It is a binary classification problem, for a given pair of questions we need to predict if they are duplicate or not.

2.2.2 Performance Metric 
Source: https://www.kaggle.com/c/quora-question-pairs#evaluation

Metric(s):

log-loss : https://www.kaggle.com/wiki/LogarithmicLoss
Binary Confusion Matrix
2.3 Train and Test Construction ¶
We build train and test by randomly splitting in the ratio of 70:30 or 80:20 whatever we choose as we have sufficient points to work with.



Feature Extraction:


Basic Features - Extracted some features before cleaning of data as below.
freq_qid1 = Frequency of qid1's
freq_qid2 = Frequency of qid2's
q1len = Length of q1
q2len = Length of q2
q1_n_words = Number of words in Question 1
q2_n_words = Number of words in Question 2
word_Common = (Number of common unique words in Question 1 and Question 2)
word_Total =(Total num of words in Question 1 + Total num of words in Question 2)
word_share = (word_common)/(word_Total)
freq_q1+freq_q2 = sum total of frequency of qid1 and qid2
freq_q1-freq_q2 = absolute difference of frequency of qid1 and qid2
Advanced Features - Did some preprocessing of texts and extracted some other features. i am giving some definitions which are used below. Token- You get a token by splitting sentence by space , Stop_Word - stop words as per NLTK, Word -A token that is not a stop_word.
cwc_min = common_word_count / (min(len(q1_words), len(q2_words))
cwc_max = common_word_count / (max(len(q1_words), len(q2_words))
csc_min = common_stop_count / (min(len(q1_stops), len(q2_stops))
csc_max = common_stop_count / (max(len(q1_stops), len(q2_stops))
ctc_min = common_token_count / (min(len(q1_tokens), len(q2_tokens))
ctc_max = common_token_count / (max(len(q1_tokens), len(q2_tokens))
last_word_eq = Check if Last word of both questions is equal or not (int(q1_tokens[-1] == q2_tokens[-1]))
first_word_eq = Check if First word of both questions is equal or not (int(q1_tokens[0] == q2_tokens[0]) )
abs_len_diff = abs(len(q1_tokens) - len(q2_tokens))
mean_len = (len(q1_tokens) + len(q2_tokens))/2
fuzz_ratio = How much percentage these two strings are similar, measured with edit distance.
fuzz_partial_ratio = if two strings are of noticeably different lengths, we are getting the score of the best matching lowest length substring.
token_sort_ratio = sorting the tokens in string and then scoring fuzz_ratio.
longest_substr_ratio = len(longest common substring) / (min(len(q1_tokens), len(q2_tokens))
Extracted Tf-Idf features for this combained question1 and question2 and got 1,2,3 gram features with Train data. Transformed test data into same vector space.
Got Word Movers Distance with pretrained glove word vectors.
From Pretrained glove word vectors got average word vector for question1 and question2. With this avg word vector got below distances.
Cosine distance
Cityblock distance
Canberra distance
Euclidean distance
Minkowski distance


Machine Learning Models:
Trained a random model to check Worst case log loss and got log loss as 0.887699
Trained some models and also tuned hyperparameters using Random and Grid search. I didnt used total train data to train my algorithms. Because of ram availability constraint in my PC, i sampled some data and Trained my models. below are models and their logloss scores. you can check total modelling and feature extraction here
For below table BF - Basic features, AF - Advanced features, DF - Distance Features including WMD.


References:
https://www.kaggle.com/c/quora-question-pairs
https://www.kaggle.com/c/quora-question-pairs/discussion
Applied AI Course
https://github.com/seatgeek/fuzzywuzzy#usage , https://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
http://proceedings.mlr.press/v37/kusnerb15.pdf
