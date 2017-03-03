import nltk
import difflib
import cosine_sim as cs
from nltk.stem.lancaster import LancasterStemmer

### Skylar labs ###
## ML/NLP/NLU Challenge ##
## By Soniya Singhal ##

## Steps:
# 1.Read training data line by line & divide it into question-answer (even odd no.)
# 2.Parse questions- tokenize, POS tagging & stemming
#	-Tokenize to extract words
#	-POS Tagging to look for NNPs (in this case)
#	-Stemming to avoid variations in statements
#	*-Stop words can be used to make sentences more general but here since questions are short, this is not done
# 3.In case POS is NNP, replace it with 'X' to make sentences general
#	-This step is taken based on the data. Trying to make training data more general as it doesn't have to depend on Pro-Nouns.
# 4.If it highly matches with any existing template questions then skip it otherwise Include it in templateQuestions.
# 5.Repeat steps 1-3 on test data.
# 6.Find the best match from template questions for every test question.
# 7.Display the answer of the template question matched.
# 8.Final accuracy of model is displayed using cosine similarity.

lancaster_stemmer = LancasterStemmer()
count=0
ans=[]
templateQs=[]
ctAns=0
k=0
### Read training file 1
with open('C:\\Users\\soniysin\\Desktop\\MaL\\Skylar Labs\\challenge-ml-nlp-nlu-master11\\challenge-ml-nlp-nlu-master\\training_dataset.txt','r')  as inFile:
    for line in inFile:
        if line is not []:
            count=count+1
            if count % 2 != 0: ##Question
                tokens = nltk.word_tokenize(line)
                posText=nltk.pos_tag(tokens)
                tok=[]
                for (w,tag) in posText:
                    w=lancaster_stemmer.stem(w)
                    if tag == 'NNP':
                        w='X'
                    tok.append(w)
                tok=" ".join(tok)
                flag=0
                for j in range(0,k):
                    if cs.get_cosine(cs.text_to_vector(tok),cs.text_to_vector(templateQs[j]))>=0.9:#Similar question already exists
                        flag=1
                        break
                if flag==0: #Add question in template Questions
                    templateQs.append(tok)
                    k=k+1
            else:
                if flag==0:
                    ans.append(line)
                    ctAns=ctAns+1

### Read training file 2
with open('C:\\Users\\soniysin\\Desktop\\MaL\\Skylar Labs\\challenge-ml-nlp-nlu-master11\\challenge-ml-nlp-nlu-master\\training_dataset_2.txt','r')  as inFile:
    for line in inFile:
        if line is not []:
            count=count+1
            if count % 2 != 0: ##Question
                tokens = nltk.word_tokenize(line)
                posText=nltk.pos_tag(tokens)
                tok=[]
                for (w,tag) in posText:
                    w=lancaster_stemmer.stem(w)
                    if tag == 'NNP':
                        w='X'
                    tok.append(w)
                tok=" ".join(tok)
                flag=0
                for j in range(0,k):
                    if cs.get_cosine(cs.text_to_vector(tok),cs.text_to_vector(templateQs[j]))>=0.9:#Similar question already exists
                        flag=1
                        break
                if flag==0: #Add question in template Questions
                    templateQs.append(tok)
                    k=k+1
            else:
                if flag==0:
                    ans.append(line)
                    ctAns=ctAns+1

tcount=0
tans=[]
tques=[]
origTest=[]
ctTans=0
### Read test file 1
with open('C:\\Users\\soniysin\\Desktop\\MaL\\Skylar Labs\\challenge-ml-nlp-nlu-master11\\challenge-ml-nlp-nlu-master\\test_dataset.txt','r') as tFile:
    for line in tFile:
        if line is not []:
            tcount=tcount+1
            if tcount % 2 !=0: ##Questions
                tokens = nltk.word_tokenize(line)
                posText=nltk.pos_tag(tokens)
                tok=[]
                for (w,tag) in posText:
                    w=lancaster_stemmer.stem(w.lower())
                    if tag == 'NNP':
                        w='X'
                    tok.append(w)
                tok=" ".join(tok)
                tques.append(tok)
                origTest.append(line)
            else:
                tans.append(line)
                ctTans=ctTans+1

### Read test file 2            
with open('C:\\Users\\soniysin\\Desktop\\MaL\\Skylar Labs\\challenge-ml-nlp-nlu-master11\\challenge-ml-nlp-nlu-master\\test_data.txt','r',encoding="utf-16") as tFile:
    for line in tFile:
        if line is not []:
            tcount=tcount+1
            if tcount % 2 !=0: ##Questions
                tokens = nltk.word_tokenize(line)
                posText=nltk.pos_tag(tokens)
                tok=[]
                for (w,tag) in posText:
                    w=lancaster_stemmer.stem(w.lower())
                    if tag == 'NNP':
                        w='X'
                    tok.append(w)
                tok=" ".join(tok)
                tques.append(tok)
                origTest.append(line)
            else:
                tans.append(line)
                ctTans=ctTans+1

sim_sum=0.0
## Predict answers for every Test question
for i in range(1,ctTans):
    sim_ques=difflib.get_close_matches(tques[i],templateQs,n=1,cutoff=0.4)
    ind=templateQs.index(sim_ques[0])
    print('TestQs:',origTest[i])
    print('PredictedAns:',ans[ind])
    print('ActualAns:',tans[i])
    sim=cs.get_cosine(cs.text_to_vector(ans[ind]),cs.text_to_vector(tans[i]))
    sim_sum+=sim
    print('Cosine similarity::',sim)
    print('*******************')

print('Accuracy of prediction model:',round(sim_sum/ctTans,3))
## ########## #############
