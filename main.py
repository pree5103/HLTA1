from collections import Counter
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
#for each review find the count for each word and add that to reviewData
for rev in reviews:
    #hashmap to store the counts of each word in the current review
    wordCounts = {}
    #translator to get rid of punctuation in rev
    translator = rev.maketrans('', '', string.punctuation.replace("'", ""))
    rev = rev.translate(translator)
    #split on space in order to get each individual word
    words = rev.split()
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
#print(unigram(0, "the"))
#print(bigram(1, "they", "were"))