from nltk.corpus import wordnet as wn
import nltk

# Download WordNet (run once)
nltk.download('wordnet')

# Function to calculate similarity
def word_similarity(word1, word2):
    syns1 = wn.synsets(word1)
    syns2 = wn.synsets(word2)
    
    if syns1 and syns2:
        return syns1[0].wup_similarity(syns2[0])
    else:
        return None

# Word pairs
word_pairs = [
    ("car", "automobile"),
    ("happy", "joyful"),
    ("king", "queen"),
    ("dog", "animal"),
    ("book", "pencil")
]

# Calculate similarity
print("---- Semantic Similarity ----\n")

for w1, w2 in word_pairs:
    sim = word_similarity(w1, w2)
    print(f"{w1} - {w2} : {sim:.2f}" if sim else f"{w1} - {w2} : No similarity found")