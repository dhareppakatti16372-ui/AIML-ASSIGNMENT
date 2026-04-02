import re

# 20 messy sentences
sentences = [
    "omg this movie was sooo goood 😂😂",
    "I cant beleive u did that!!",
    "lol that was funny af 🤣",
    "gonna go now ttyl",
    "this is amazng bro!!!",
    "idk what u mean 🤔",
    "heyyy wassup??",
    "dats cool 😎",
    "I luv this song sooo much",
    "brb need to chk something",
    "why r u late???",
    "haha that meme tho 😂",
    "im sooo tired rn 😴",
    "wt r u doing??",
    "this is gr8 work 👍",
    "omg cant stop laughing 🤣🤣",
    "pls send me d details asap",
    "u r amazing!!!",
    "that was lit 🔥🔥",
    "sry for the delay bro"
]

# slang dictionary
slang_dict = {
    "omg": "oh my god",
    "lol": "laughing out loud",
    "ttyl": "talk to you later",
    "idk": "i do not know",
    "brb": "be right back",
    "rn": "right now",
    "gr8": "great",
    "pls": "please",
    "asap": "as soon as possible",
    "sry": "sorry",
    "u": "you",
    "r": "are",
    "d": "the",
    "wt": "what",
    "luv": "love"
}

# remove emojis
def remove_emojis(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# reduce repeated letters (sooo -> soo)
def reduce_repeated_chars(text):
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

# expand slang
def expand_slang(text):
    words = text.split()
    return " ".join([slang_dict.get(word, word) for word in words])

# remove punctuation
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

# full preprocessing function
def preprocess(text):
    text = text.lower()
    text = remove_emojis(text)
    text = expand_slang(text)
    text = reduce_repeated_chars(text)
    text = remove_punctuation(text)
    return text

# apply preprocessing
print("---- Cleaned Sentences ----\n")
for i, sentence in enumerate(sentences, 1):
    cleaned = preprocess(sentence)
    print(f"{i}. {cleaned}")