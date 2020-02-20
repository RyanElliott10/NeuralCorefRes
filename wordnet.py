from nltk.corpus import wordnet as wn

# syn = wn.synsets('dog')[0]
# print(syn.hyponyms())

# nltk.download('omw')

human_synset = wn.synsets('human')[0]
dog_synset = wn.synsets('dog')[0]
cat_synset = wn.synsets('cat')[0]
pineapple_synset = wn.synsets('pineapple')[0]
canine_synset = wn.synsets('canine')[0]
mutt_synset = wn.synsets('mutt')[0]

print(dog_synset)
print(cat_synset)
print(pineapple_synset)

print(f"Dog to dog: {dog_synset.wup_similarity(dog_synset)}")
print(f"Pineapple to cat: {pineapple_synset.wup_similarity(cat_synset)}")
print(f"Cat to dog: {cat_synset.wup_similarity(dog_synset)}")
print(f"Dog to human: {dog_synset.wup_similarity(human_synset)}")
print(f"Dog to pineapple: {dog_synset.wup_similarity(pineapple_synset)}")
print(f"Dog to canine: {dog_synset.wup_similarity(canine_synset)}")
print(f"Dog to mutt: {dog_synset.wup_similarity(mutt_synset)}")

# Synsets = wn.synsets('dog', pos=wn.NOUN)
# print(Synsets)

# for Synset in Synsets:
#   print(Synset)
#   print("DEFINITION: ",Synset.definition())
#   for example in Synset.examples():
#     print("\tEXAMPLE:",example)
#     words = []
#     for lemma in Synset.lemmas():
#       words.append(lemma.name())

#       hypoWords = []
#       for hypo in Synset.hyponyms():
#         for lemma in hypo.lemmas():
#           hypoWords.append(lemma.name())

#       hyperWords = []
#       for hyper in Synset.hypernyms():
#         for lemma in hyper.lemmas():
#           hyperWords.append(lemma.name())

#     print("\tLEMMAS: ",", ".join(words))
#     print("\tHYPONYMS: ",", ".join(hypoWords))
#     print("\tHYPERNYMS: ",", ".join(hyperWords))
#     print()