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
def build_vocab(reviews, fixed_vocab_size=None, min_freq=2):
    word_ct = Counter(word for rev in reviews for word in rev.split())
    if fixed_vocab_size: #fixed vocab: keep top V words
        most_common_words = {word for word, _ in word_ct.most_common(fixed_vocab_size)}
    else: #implicit vocab: keep words appearing at least min_freq times
        most_common_words = {word for word, count in word_ct.items() if count >= min_freq}
    most_common_words.add("<UNK>") #add unknown word token
    return most_common_words

#applying fixed or implicit vocab
fixed_vocab = build_vocab(reviews, fixed_vocab_size=3000)
implicit_vocab = build_vocab(reviews, min_freq=2)

print("\n===== DEBUG: Vocabulary Check =====")
print("'they' in Implicit Vocab?:", "they" in implicit_vocab)
print("'were' in Implicit Vocab?:", "were" in implicit_vocab)

#holds all the count data for each unigram of the review
reviewData = {}
revNum = 0 #the current review number

#choose whether to use fixed or implicit vocab
use_fixed_vocab = False #change to False to use implicit vocab

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
    if use_fixed_vocab:
        words = [word if word in fixed_vocab else "<UNK>" for word in words]
    else:
        words = [word if word in implicit_vocab else "<UNK>" for word in words]

    # Store the updated words back in `reviews` so bigram() also gets them
    reviews[revNum] = " ".join(words)  # Replacing the original review text
    
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
    return round(prob, 2) if totalWords > 0 else 0.0


#BIGRAM
def bigram(revNum, word1, word2):
    review = reviews[revNum]
    words = review.split()
    
    # Replace unknown words with <UNK>
    vocab = fixed_vocab if use_fixed_vocab else implicit_vocab
    words = [word if word in vocab else "<UNK>" for word in words]
    
    total_pairs = len(words) - 1
    bigram_count = sum(1 for i in range(total_pairs) if words[i] == word1 and words[i+1] == word2)
    
    probability = bigram_count / total_pairs if total_pairs > 0 else 0.0
    
    # print(f"Debug - Review {revNum}: {' '.join(words[:20])}...") # Print first 20 words
    # print(f"Debug - Bigram count for '{word1} {word2}': {bigram_count}")
    # print(f"Debug - Total pairs: {total_pairs}")
    # print(f"Debug - <UNK> count: {words.count('<UNK>')}")
    
    return round(probability, 2)


 
#test unsmoothed
print(unigram(0, "the"))
print(bigram(1, "they", "were"))

#test handling unknown words
print("unigram prob of <UNK>:", unigram(0, "<UNK>"))  # Check if <UNK> is handled correctly on unigram
print("bigram prob of <UNK> and they:", bigram(1, "<UNK>", "they"))  # Check if <UNK> is handled correctly on bigram
print("bigram prob of they and <UNK>:", bigram(1, "they", "<UNK>"))
print("bigram prob of <UNK> and <UNK>:", bigram(1, "<UNK>", "<UNK>"))
#more test cases to check different scenarios
print("Unigram prob of 'the':", unigram(0, "the"))
print("Bigram prob of 'they were':", bigram(1, "they", "were"))
print("Unigram prob of <UNK>:", unigram(0, "<UNK>"))
print("Bigram prob of <UNK> and 'the':", bigram(0, "<UNK>", "the"))
print("Bigram prob of 'the' and <UNK>:", bigram(0, "the", "<UNK>"))
print("Bigram prob of <UNK> and <UNK>:", bigram(0, "<UNK>", "<UNK>"))
