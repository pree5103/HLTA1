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
    translator = rev.maketrans('', '', string.punctuation)
    rev = rev.translate(translator)
    #split on space in order to get each individual word
    words = rev.split()
    #holds the word counts for each word in the review rev
    wordCounts = Counter(words)
    #set the reviewData for that review to wordCounts
    reviewData[revNum] = wordCounts
    # increment revNum
    revNum += 1


