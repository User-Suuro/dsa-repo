import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pathlib import Path

# word_tokenize accepts a string as an input, not a file.
stop_words = set(stopwords.words('english'))

folder = Path(__file__).parent

file1 = open(folder / "text.txt")

# Use this to read file content as a stream:
line = file1.read()

words = line.split()

for r in words:
    if r not in stop_words:
        appendFile = open(folder / "filteredtext.txt", 'a')
        appendFile.write(" " + r)
        appendFile.close()