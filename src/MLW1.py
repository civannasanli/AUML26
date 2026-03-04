from hmmlearn.hmm import CategoricalHMM
import numpy as np
#ev kelimesi için model
model_ev = CategoricalHMM(n_components=2)
model_ev.startprob_=np.array([1.0,0.0])
####################
transmat_ev = np.array([[0.6,0.4],[0.2,0.8]])
emission_ev = np.array([[0.7,0.3],[0.1,0.9]])
####################
model_ev.transmat_ = transmat_ev
model_ev.emissionprob_ = emission_ev
model_ev.n_features = 2
#####################
model_okul = CategoricalHMM(n_components=4)
model_okul.startprob_=np.array([1.0,0.0,0.0,0.0])
#####################
transmat_okul = np.array([[0.3,0.4,0.1,0.2],[0.1,0.5,0.2,0.2],[0.2,0.3,0.4,0.1],[0.1,0.2,0.3,0.4]])
emission_okul = np.array([[0.6,0.4],[0.3,0.7],[0.2,0.8],[0.1,0.9]])
#####################
model_okul.transmat_ = transmat_okul
model_okul.emissionprob_ = emission_okul
model_okul.n_features = 2
#################TESTDATA
testdata0 = np.array([[0],[1],[0],[0]])
testdata1 = np.array([[1],[0],[1],[1]])
testdata2 = np.array([[0],[1]])
testdata3 = np.array([[1],[0]])
#################
score_ev = model_ev.score(testdata0)
score_okul = model_okul.score(testdata0)
if score_ev > score_okul:
    print("ev")
else:    print("okul")
score_ev = model_ev.score(testdata1)
score_okul = model_okul.score(testdata1)
if score_ev > score_okul:
    print("ev")
else:    print("okul")
score_ev = model_ev.score(testdata2)
score_okul = model_okul.score(testdata2)
if score_ev > score_okul:
    print("ev")
else:    print("okul")
score_ev = model_ev.score(testdata3)
score_okul = model_okul.score(testdata3)
if score_ev > score_okul:
    print("ev")
else:    print("okul")

