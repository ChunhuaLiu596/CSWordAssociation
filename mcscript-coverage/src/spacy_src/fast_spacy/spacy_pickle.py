import spacy
import pickle
nlp = spacy.load('en_core_web_sm')
texts = ["Hello friend", "how is your day?", "I wish you all the best"]
docs = [nlp(text) for text in texts]

''' Serialization '''
# Serialize vocab (actually the whole NLP ojbect)
pickle.dump(nlp, open("data/debug/Obj.pickle", "wb"))
pickle.dump(docs, open("data/debug/docs.pickle", "wb"))

''' Deserialization '''
nlp = pickle.load(open("data/debug/Obj.pickle", "rb"))
docs = pickle.load(open("data/debug/docs.pickle", "rb"))