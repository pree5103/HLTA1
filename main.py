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
#the current review number 
revNum = 0
# Number of bigrams for use in smoothing
bigram_counts = defaultdict(Counter)
unigram_counts = Counter()
#vocab_size = set()  # No longer need to track vocabulary size as a set because of the build vocab
#choose whether to use fixed or implicit vocab
use_fixed_vocab = False #change to False to use implicit vocab

#for each review find the count for each word and add that to reviewData
for rev in reviews:
    # translator to get rid of punctuation in rev
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
    unigram_counts.update(words)
    # Track vocab size as a set to get unique words (equivalent to vocab size)
    #vocab_size.update(words) #No longer updating since we got the build vocab function
    #set the reviewData for that review to wordCounts
    reviewData[revNum] = wordCounts
    # increment revNum
    revNum += 1
    # Update bigram counts (aka word combinations)
    for i in range(len(words) - 1):
        bigram_counts[words[i]][words[i + 1]] += 1
        
vocab = fixed_vocab if use_fixed_vocab else implicit_vocab
vocab_size = len(vocab)
#print(len(vocab_size))
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

def laplacian_smoothing(word1, word2):
    # Replace unknown words with <UNK>
    if word1 not in vocab:
        word1 = "<UNK>"
    if word2 not in vocab:
        word2 = "<UNK>"
        
    # Fetch our bigram count for this pair of words, unigram count for word1, and vocab size
    bigram_count = bigram_counts[word1].get(word2, 0)
    unigram_count = unigram_counts.get(word1, 0)

    # Just to avoid 0 division (though it's only really a problem if the unigram_count is 0)
    if unigram_count == 0:
        return 0.0
    
    # We have all of our info, now it's just applying the formula we had from class (bigram count + 1 divided by sum of word1 count and vocab size)
    probability = (bigram_count + 1) / (unigram_count + vocab_size)
    return probability

# Adjust k to whatever you want, I just left 1 to be our placeholder
def add_k_smoothing(word1, word2, k=1):
    # Replace unknown words with <UNK>
    if word1 not in vocab:
        word1 = "<UNK>"
    if word2 not in vocab:
        word2 = "<UNK>"
    # Same set-up as Laplacian
    bigram_count = bigram_counts[word1].get(word2, 0)
    unigram_count = unigram_counts.get(word1, 0)
    
    if unigram_count == 0:
        return 0.0
    
    # Slightly different formula where k is added to bigram count and multiplied by vocab size in denominator
    probability = (bigram_count + k) / (unigram_count + (k * vocab_size))
    return probability

#test unsmoothed
print(unigram(0, "the"))
print(bigram(1, "they", "were"))

#test handling unknown words
print("unigram prob of <UNK>:", unigram(0, "<UNK>"))  # Check if <UNK> is handled correctly on unigram
print("bigram prob of <UNK> and they:", bigram(1, "<UNK>", "they"))  # Check if <UNK> is handled correctly on bigram
print("bigram prob of they and <UNK>:", bigram(1, "they", "<UNK>"))
print("bigram prob of <UNK> and <UNK>:", bigram(1, "<UNK>", "<UNK>"))
#more test cases to check different scenarios
print("unigram prob of 'the':", unigram(0, "the"))
print("bigram prob of 'they were':", bigram(1, "they", "were"))
print("unigram prob of <UNK>:", unigram(0, "<UNK>"))
print("bigram prob of <UNK> and 'the':", bigram(0, "<UNK>", "the"))
print("bigram prob of 'the' and <UNK>:", bigram(0, "the", "<UNK>"))
print("bigram prob of <UNK> and <UNK>:", bigram(0, "<UNK>", "<UNK>"))

#test the smoothing methods
print("Laplacian prob: ", laplacian_smoothing("they", "were"))
print("Add K prob: ", add_k_smoothing("they", "were"))
print("Laplacian prob with <UNK>: ", laplacian_smoothing("<UNK>", "the"))
print("Add K prob with <UNK>: ", add_k_smoothing("<UNK>", "the"))
print("Laplacian prob with unknown word: ", laplacian_smoothing("zzzz", "the"))
print("Add K prob with unknown word: ", add_k_smoothing("zzzz", "the"))
