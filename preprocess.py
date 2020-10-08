import re
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from emoji.unicode_codes import UNICODE_EMOJI

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('wordnet')
lmtzr = WordNetLemmatizer() 

def stopWords(tweet):
  return " ".join([word for word in tweet.split() if word not in stopwords.words('english')])

def lemmatize(tweet): 
  return ' '.join([lmtzr.lemmatize(word, 'v') for word in tweet.split()])

def lower(tweet):
  return tweet.lower()

def emoji(tweet):
  out = []
  for word in tweet.split():
    if word in UNICODE_EMOJI:
      word = UNICODE_EMOJI[word]
    out.append(word)
  return ' '.join(out)

def charSqueeze(tweet):
  squeezed = []
  prev = None
  rep = False
  for curr_char in tweet:
    if rep:
      if curr_char != prev:
        rep = False 
        squeezed.append(curr_char)
    else:
      squeezed.append(curr_char)
      if prev == curr_char:
        rep = True
      else:
        prev = curr_char
  return ''.join(squeezed)
  # return re.sub(r'([A-z])(?=[A-z]\1)', "", tweet)

def rmurls(tweet):
  return re.sub("HTTPURL", "", tweet)

def rmUser(tweet):
  return re.sub("@USER", "", tweet)

def AlNum(tweet):
  clean = []
  for word in tweet.split():
    if word[0] == '#' or word[0] == ':' or str.isalnum(word):
      clean.append(word)
    else:
      clean.append(re.sub(r'[\W_]+', '', word, flags=re.UNICODE))
  return ' '.join(clean)

def hashtag(tweet):
  clean = []
  for word in tweet.split():
    if word[0] == '#':
      word = word[1:]
    clean.append(word)
  return ' '.join(clean)

def process(corpora, ur=True, us=True, ha=True, ch=True, em=False, an=False, sw=False, lo=False, le=False):
  clean_corpora = []
  for tweet in corpora:
    if ur:
      tweet = rmurls(tweet)
    if us:
      tweet = rmUser(tweet)
    if sw:
      tweet = stopWords(tweet)
    if ch:
      tweet = charSqueeze(tweet)
    if em:
      tweet = emoji(tweet)
    if lo:
      tweet = lower(tweet)
    if le:
      tweet = lemmatize(tweet)
    if an:
      tweet = AlNum(tweet)
    if ha:
      tweet = hashtag(tweet)
    clean_corpora.append(tweet)
  return clean_corpora
