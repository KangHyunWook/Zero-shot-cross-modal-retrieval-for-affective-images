# First, you're going to need to import wordnet:
from nltk.corpus import wordnet
  
# Then, we're going to use the term "program" to find synsets like so:
syns = wordnet.synsets("sad")

print(len(syns))

for i in range(len(syns)):
    print(syns[i].lemmas()[0].name())
# print(syns[0].name())
  
# # Just the word:
# print(syns[2].lemmas()[0].name())
  
# # # Definition of that first synset:
# print(syns[0].definition())
  
# # Examples of the word in use in sentences:
# print(syns[0].examples())