
# The following program removes stop words from a piece of text:

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


example_sent =  """This is a sample sentence, showing off the stop words filtration."""

stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(example_sent)

# Converts the words in word_tokens to lower case and then checks
# whether they are present in stop_words or not
filtered_sentence = [w for w in word_tokens if w.lower() not in stop_words]

# With no lower-case conversion
filtered_sentence = []

for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

print(word_tokens)
print(filtered_sentence)
