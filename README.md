# NLP-Challenge
### Skylar labs ###
## ML/NLP/NLU Challenge ##
## By Soniya Singhal ##



Coding is in Python making use of NLTK.
## Steps:
# 1.Read training data line by line & divide it into question-answer (even odd no.)
# 2.Parse questions- tokenize, POS tagging & stemming
	-Tokenize to extract words
	-POS Tagging to look for NNPs (in this case)
	-Stemming to avoid variations in statements
	*-Stop words can be used to make sentences more general but here since questions are short, this is not done
# 3.In case POS is NNP, replace it with 'X' to make sentences general
	-This step is taken based on the data. Trying to make training data more general as it doesn't have to depend on Pro-Nouns.
# 4.If it highly matches with any existing template questions then skip it otherwise Include it in templateQuestions.
# 5.Repeat steps 1-3 on test data.
# 6.Find the best match from template questions for every test question.
# 7.Display the answer of the template question matched.
# 8.Final accuracy of model is displayed using cosine similarity.


Run skylar_labs.py.
Note:cosine_sim.py is helper file.
Model accuracy of 89.90% is obtained.
