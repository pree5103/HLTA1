from collections import Counter
import string

# DATA NORMALIZATION FOR EACH REVIEW

#opening the train file 
file = open("train.txt", "r")
reviews = [] #holds all the reviews

#populates reviews with the reviews in file
for i in file:
    #makes every character in i, lowercase
    reviews.append(i.lower())

#building vocabulary 
def build_vocab(reviews, fixed_vocab_size=None, min_freq=2)
    word_ct = Counter(word for rev in reviews for word in rev.split())
    if fixed_vocab_size: #fixed vocab: keep top V words
	most_common_words = {word for word, _ in word_counts.most_common(fixed_vocab_size)}
    else: #implicit vocab: keep words appearing at least min_freq times
	most_common_words = {word for word, count in word_counts.items() if count >= min_freq}
    return most_common_words
#applying fixed or implicit vocab
fixed_vocab = build_vocab(reviews, fixed_vocab_size=50000)
implicit_vocab = build_vocab(reviews, min_freq=2)


#holds all the count data for each unigram of the review
reviewData = {}
 
revNum = 0 #the current review number

#for each review find the count for each word and add that to reviewData
for rev in reviews:
    #hashmap to store the counts of each word in the current review
    wordCounts = {}

    #translator to get rid of punctuation in rev
    translator = rev.maketrans('', '', string.punctuation.replace("'", ""))
    rev = rev.translate(translator)

    #split on space in order to get each individual word
    words = rev.split()

    #replace unknown words
    # words = [word if word in fixed_vocab else "<UNK>" for word in  words]
    # words = [word if word in implicit_vocab else "<UNK>" for word in  words]

    #holds the word counts for each word in the review rev
    wordCounts = Counter(words)

    #set the reviewData for that review to wordCounts
    reviewData[revNum] = wordCounts
    # increment revNum
    revNum += 1

# UNIGRAM 
def unigram(revNum, word):
    #holds the data for the specific review we're analyzing
    review = reviewData[revNum]

    #the total words in the review
    totalWords = sum(review.values())

    #the number of occurrences of the word in the review
    #also handling unknown words within
    wordCount = review.get(word, review.get("<UNK>", 0))

    #the probability of the unigram
    prob = wordCount / totalWords

    #return the probability rounded to 2 digits
    return round(prob, 2)if totalWords > 0 else 0.0

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
#test unsmoothed
#print(unigram(0, "the"))
#print(bigram(1, "they", "were"))

#test handling unknown words
#print(unigram(0, "<UNK>"))  # Check if <UNK> is handled correctly on unigram
