# Preprocess two texts 'text1' and 'text2'
print('\nQuestion 5')
with open('week6/text1', 'r') as file1:
    content1 = file1.read().lower()
with open('week6/text2', 'r') as file2:
    content2 = file2.read().lower()

content = [content1, content2]


#vectorize the two texts using the CountVectorizer. Using n-gram size of 1-3 for the vectorizer.
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(1,3))
X = vectorizer.fit_transform(content)

# Compute the cosine similarity between the texts
from sklearn.metrics.pairwise import cosine_similarity

cosineSim = cosine_similarity(X)
print(f"Cosine Similarity is: {cosineSim}")

# Compute the jaccard similarity between text1 and text2. This is a statistical measure to gauge similarity and diversity in text by looking at 
#overlap in words or tokens. More specifically it is "the ratio of the size of the intersection of two sets to the size of the union of those sets"
#1 would mean they are identical, 0 that they have no elements in common
set1 = set(content1.split())
set2 = set(content2.split())

intersection = set1.intersection(set2)
union = set1.union(set2)

jaccardSim = len(intersection) / len(union)
print(f"\nJaccard Similarity: {jaccardSim}")


# Train a POS tagger.
from nltk.corpus import brown
from nltk.tag import UnigramTagger, BigramTagger, DefaultTagger
import pickle
import nltk
nltk.download('brown')
nltk.download('punkt_tab')

brownTaggedSents = brown.tagged_sents(categories='news')
# Use 90% of the data for training and 10% for evaluation.
split = int(0.9 * len(brownTaggedSents))
trainingData = brownTaggedSents[:split]
testData = brownTaggedSents[split:]

# Comparing the performances of unigram, bigram and bigram with unigram as backoff
defaultTagger = DefaultTagger('NN')
unigramTagger = UnigramTagger(trainingData)
bigramTagger = BigramTagger(trainingData)
combineTagger = BigramTagger(trainingData, backoff=unigramTagger)

# Comparing performance of different taggers
unigramAccuracy = unigramTagger.evaluate(testData)
bigramAccuracy = bigramTagger.evaluate(testData)
combineAccuracy = combineTagger.evaluate(testData)

print(f'Accuracy of Unigram: {unigramAccuracy}')
print(f'Accuracy of Bigram: {bigramAccuracy}')
print(f'Accuracy of combined: {combineAccuracy}')

# Save and load the tagger.
with open('combineTagger.pkl', 'wb') as f:
    pickle.dump(combineTagger, f)

with open('combineTagger.pkl', 'rb') as f:
    loadTagger = pickle.load(f)

# Tag all the words from text1.
with open('week6/text1', 'r') as f:
    words = f.read().lower()
    wordsToken = nltk.word_tokenize(words)

taggedWords = loadTagger.tag(wordsToken)

print(taggedWords)
