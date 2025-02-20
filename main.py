from collections import Counter, defaultdict
import string

# DATA NORMALIZATION FOR EACH REVIEW

#opening the train file 
file = open("train.txt", "r")
#holds all the reviews
reviews = []
#populates reviews with the reviews in file
for i in file:
    #makes every character in i, lowercase
    reviews.append(i.lower())
#holds all the count data for each unigram of the review
reviewData = {}
#the current review number 
revNum = 0
# Number of bigrams for use in smoothing
bigram_counts = defaultdict(Counter)
unigram_counts = Counter()
vocab_size = set()
#for each review find the count for each word and add that to reviewData
for rev in reviews:
    # translator to get rid of punctuation in rev
    translator = rev.maketrans('', '', string.punctuation.replace("'", ""))
    rev = rev.translate(translator)
    #split on space in order to get each individual word
    words = rev.split()
    #holds the word counts for each word in the review rev
    wordCounts = Counter(words)
    unigram_counts.update(words)
    # Track vocab size as a set to get unique words (equivalent to vocab size)
    vocab_size.update(words)
    #set the reviewData for that review to wordCounts
    reviewData[revNum] = wordCounts
    # increment revNum
    revNum += 1
    # Update bigram counts (aka word combinations)
    for i in range(len(words) - 1):
        bigram_counts[words[i]][words[i + 1]] += 1
#print(len(vocab_size))
# UNIGRAM 
def unigram(revNum, word):
    #holds the data for the specific review we're analyzing
    review = reviewData[revNum]
    #the total words in the review
    totalWords = sum(review.values())
    #the number of occurences of the word in the review
    wordCount = review[word]
    #the probability of the unigram
    prob = wordCount / totalWords
    #return the probability rounded to 2 digits
    return round(prob, 2)

#BIGRAM
def bigram(revNum, word1, word2):
    #populate review with the specified review
    review = reviews[revNum]
    # translator to get rid of punctuation in the review
    translator = review.maketrans('', '', string.punctuation.replace("'", ""))
    review = review.translate(translator)
    # split the review to get the words in order
    review = review.split()
    #count of the bigrams where word2 follows word1
    correctPairs = 0
    #count of the bigrams where word1 is the first word in the word pair
    wordPairs = 0
    #loop through all of the words in the review
    for i in range(len(review) - 1):
        #if a correct word pair is found, increment correctPairs and wordPairs
        if review[i] == word1 and review[i + 1] == word2:
            correctPairs += 1
            wordPairs += 1
        #if a word pair where word1 is the first word is found, increment wordPairs
        elif review[i] == word1 and review[i + 1] != word2:
            wordPairs += 1
    #probability is the percentage of correctPairs in all of the found wordPairs
    probability = correctPairs / wordPairs
    return probability

def laplacian_smoothing(word1, word2):
    # Fetch our bigram count for this pair of words, unigram count for word1, and vocab size
    bigram_count = bigram_counts[word1].get(word2, 0)
    unigram_count = unigram_counts.get(word1, 0)
    v_size = len(vocab_size)

    # Just to avoid 0 division (though it's only really a problem if the v_size is also 0)
    if unigram_count == 0:
        return 0.0
    
    # We have all of our info, now it's just applying the formula we had from class (bigram count + 1 divided by sum of word1 count and vocab size)
    probability = (bigram_count + 1) / (unigram_count + v_size)
    return probability

# Adjust k to whatever you want, I just left 1 to be our placeholder
def add_k_smoothing(word1, word2, k=1):
    # Same set-up as Laplacian
    bigram_count = bigram_counts[word1].get(word2, 0)
    unigram_count = unigram_counts.get(word1, 0)
    v_size = len(vocab_size)
    
    if unigram_count == 0:
        return 0.0
    
    # Slightly different formula where k is added to bigram count and multiplied by vocab size in denominator
    probability = (bigram_count + k) / (unigram_count + (k * v_size))
    return probability

#print("Unigram Probability: ", unigram(0, "the"))
#print("Bigram Probability: ", bigram(1, "they", "were"))
#print("Laplacian Probability: ", laplacian_smoothing("they", "were"))
#print("Add K Probability: ", add_k_smoothing("they", "were"))
