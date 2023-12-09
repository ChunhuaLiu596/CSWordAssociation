import spacy 

nlp = spacy.load('en_core_web_sm')

doc1 = nlp("Hello world")
doc2 = nlp("This is a test")

# doc1_data = pickle.dumps(doc1)
# doc2_data = pickle.dumps(doc2)
# print(len(doc1_data) + len(doc2_data))  # 6636116 😞

doc_data = pickle.dumps([doc1, doc2])
print(len(doc_data))  # 3319761 😃