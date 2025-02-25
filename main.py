from collections import Counter, defaultdict
import string
import math

# DATA NORMALIZATION FOR EACH REVIEW

#opening the train file 
file = open("train.txt", "r")
#holds all the reviews
reviews = []
#populates reviews with the reviews in file
for i in file:
    #makes every character in i lowercase
    reviews.append(i.lower())

#preprocessing function to remove punctuation and tokenize
#replaced old preprocessing within processing reviews loop with a centralized function to avoid inconsistencies between review processing and vocab building
def preprocess_line(line):
    translator = line.maketrans('', '', string.punctuation.replace("'", ""))
    line = line.translate(translator).strip()
    return line.split()

#building vocabulary with unknown handling
def build_vocab(reviews, fixed_vocab_size=None, min_freq=2):
    word_ct = Counter(word for rev in reviews for word in preprocess_line(rev)) #use standardized preprocessing function
    if fixed_vocab_size:  #fixed vocab: keep top V words
        most_common_words = {word for word, _ in word_ct.most_common(fixed_vocab_size)}
    else:  #implicit vocab: keep words appearing at least min_freq times
        most_common_words = {word for word, count in word_ct.items() if count >= min_freq}
    most_common_words.update(["<UNK>", "<s>", "</s>"])  #add unknown word token and sentence boundaries
    return most_common_words

#declare vocabularies: fixed (top 3000 words) or implicit (words with >= 2 occurrences)
fixed_vocab = build_vocab(reviews, fixed_vocab_size=3000)
implicit_vocab = build_vocab(reviews, min_freq=2)

#function to replace unknown words
def replace_unknown_words(tokens, vocab):
    return [word if word in vocab else "<UNK>" for word in tokens]

#choose whether to use fixed or implicit vocab
use_fixed_vocab = False  #set to True to use fixed vocab
vocab = fixed_vocab if use_fixed_vocab else implicit_vocab
vocab_size = len(vocab)

#holds unigram and bigram counts
unigram_counts = Counter()
bigram_counts = defaultdict(Counter)

#processing reviews for n-gram counts
for revNum, rev in enumerate(reviews):
    words = preprocess_line(rev) #use standardized preprocessing function
    words = replace_unknown_words(words, vocab)
    words = ["<s>"] + words + ["</s>"]  #add sentence boundaries
    
    unigram_counts.update(words)
    for i in range(len(words) - 1):
        bigram_counts[words[i]][words[i + 1]] += 1
    
    reviews[revNum] = " ".join(words)  #replace original review text with processed version

#unigram - unsmoothed
def unigram(revNum, word):
    word = word if word in vocab else "<UNK>" #checks if it's a known word in the vocab
    totalWords = sum(unigram_counts.values())
    prob = unigram_counts[word] / totalWords if totalWords > 0 else 0.0
    return round(prob, 4)

#bigram - unsmoothed
def bigram(revNum, word1, word2):
    word1, word2 = replace_unknown_words([word1, word2], vocab) #ensures any OOV words are replaced with <UNK> 
    total_pairs = sum(bigram_counts[word1].values())
    probability = bigram_counts[word1].get(word2, 0) / total_pairs if total_pairs > 0 else 0.0
    return round(probability, 4)

#laplacian smoothing function
def laplacian_smoothing(word1, word2):
    #replace unknown words with <UNK>
    word1, word2 = replace_unknown_words([word1, word2], vocab)
    #fetch our bigram count for this pair of words, unigram count for word1, and vocab size
    bigram_count = bigram_counts[word1].get(word2, 0)
    unigram_count = unigram_counts.get(word1, 0)
    #just to avoid 0 division (though it's only really a problem if the unigram_count is 0)
    if unigram_count == 0:
        return 0.0
    #we have all of our info, now it's just applying the formula we had from class (bigram count + 1 divided by sum of word1 count and vocab size)
    probability = (bigram_count + 1) / (unigram_count + vocab_size)
    return round(probability, 4) #round up to 4 digits 

#add-K smoothing function
def add_k_smoothing(word1, word2, k=1):
    #replace unknown words with <UNK>
    word1, word2 = replace_unknown_words([word1, word2], vocab)
    #same setup as Laplacian
    bigram_count = bigram_counts[word1].get(word2, 0)
    unigram_count = unigram_counts.get(word1, 0)
    if unigram_count == 0:
        return 0.0
    #slightly different formula where k is added to bigram count and multiplied by vocab size in denominator
    probability = (bigram_count + k) / (unigram_count + (k * vocab_size))
    return round(probability, 4)

#test cases
print("===================================")
print("IMPLICIT VOCAB (UNSMOOTHED)")
vocab = implicit_vocab  #ensure implicit vocab is used
print("Unigram prob of 'the':", unigram(0, "the")) #check seen words
print("Unigram prob of <UNK>:", unigram(0, "<UNK>")) #check UNK
#0 is safe to used because we need a baseline to verify if unknown words are handled correctly
print("Bigram prob of 'they' 'were':", bigram(1, "they", "were")) #check seen words
print("Bigram prob of <UNK> and 'the':", bigram(0, "<UNK>", "the")) #check UNK
print("===================================")
print("FIXED VOCABULARY (UNSMOOTHED)")
vocab = implicit_vocab  #ensure implicit vocab is used
print("Unigram prob of 'the':", unigram(0, "the")) #check seen words
print("Unigram prob of <UNK>:", unigram(0, "<UNK>")) #check UNK
#0 is safe to used because we need a baseline to verify if unknown words are handled correctly
print("Bigram prob of 'they' 'were':", bigram(1, "they", "were")) #check seen words
print("Bigram prob of <UNK> and 'the':", bigram(0, "<UNK>", "the")) #check UNK
print("===================================")
print("LAPLACIAN SMOOTHING")
vocab = implicit_vocab
print("Laplacian Probability with implicit vocab: ", laplacian_smoothing("zzzz", "the"))
vocab = fixed_vocab
print("Laplacian Probability with fixed vocab: ", laplacian_smoothing("zzzz", "the"))
print("===================================")
print("ADD-K SMOOTHING")
vocab = implicit_vocab
print("Add-K Probability with implicit vocab: ", add_k_smoothing("zzzz", "the"))
vocab = fixed_vocab
print("Add-K Probability with fixed vocab: ", add_k_smoothing("zzzz", "the"))
print("===================================")

#debug: in case probabilities are identical between implicit and fixed vocabularies
# print("Size of implicit vocab:", len(implicit_vocab))
# print("Size of fixed vocab:", len(fixed_vocab))
# percent_diff = (len(implicit_vocab) - len(fixed_vocab)) / len(implicit_vocab) * 100
# print(f"Percentage difference: {percent_diff:.2f}%")
# if percent_diff <= 5: #5% is the threshold 
#    print("Less than 5%. No action required.")
# else:
#    print("More than 5%. Check for discrepancies between implicit and fixed vocabularies.")

#opening the validation file and converting all text to lowercase
with open("val.txt", "r") as f:
    val_reviews = [line.lower() for line in f]

#function to compute unigram perplexity
def compute_unigram_perplexity(val_reviews):
    total_log_prob = 0.0  #keeps track of total log probability
    total_tokens = 0  #counts total tokens in validation set
    total_words = sum(unigram_counts.values())  #total words in training data

    for line in val_reviews:
        tokens = replace_unknown_words(preprocess_line(line), vocab)  #preprocess validation text
        for token in tokens:
            p = unigram_counts.get(token, 0) / total_words if total_words > 0 else 0  #unigram probability
            if p == 0:
                p = 1e-10  #avoid log(0) issue
            total_log_prob += math.log(p)
            total_tokens += 1

    avg_log_prob = total_log_prob / total_tokens if total_tokens > 0 else float('-inf')  #compute average log probability
    return math.exp(-avg_log_prob)  #compute perplexity

#function to compute unsmoothed bigram probability
def unsmoothed_bigram_prob(word1, word2):
    word1, word2 = replace_unknown_words([word1, word2], vocab)  #replace unknown words with <unk>
    total_pairs = sum(bigram_counts[word1].values())  #total occurrences of word1 in bigram model
    p = bigram_counts[word1].get(word2, 0) / total_pairs if total_pairs > 0 else 0.0  #bigram probability
    return p

#function to compute bigram perplexity
def compute_bigram_perplexity(val_reviews, smoothing_func):
    total_log_prob = 0.0  #keeps track of total log probability
    total_tokens = 0  #counts total tokens in validation set

    for line in val_reviews:
        tokens = replace_unknown_words(preprocess_line(line), vocab)  #preprocess validation text
        tokens = ["<s>"] + tokens + ["</s>"]  #add sentence boundary tokens

        for i in range(len(tokens)-1):
            p = smoothing_func(tokens[i], tokens[i+1])  #compute probability using smoothing function
            if p == 0:
                p = 1e-10  #avoid log(0) issue
            total_log_prob += math.log(p)
            total_tokens += 1

    avg_log_prob = total_log_prob / total_tokens if total_tokens > 0 else float('-inf')  #compute average log probability
    return math.exp(-avg_log_prob)  #compute perplexity

#print perplexity results for different models
print("PERPLEXITY")
print("Unigram Perplexity:", round(compute_unigram_perplexity(val_reviews),4))
print("Bigram Perplexity (Unsmoothed):", round(compute_bigram_perplexity(val_reviews, unsmoothed_bigram_prob),4))
print("Bigram Perplexity (Laplacian):", round(compute_bigram_perplexity(val_reviews, laplacian_smoothing),4))
print("Bigram Perplexity (Add-k, K=1):", round(compute_bigram_perplexity(val_reviews, lambda w1, w2: add_k_smoothing(w1, w2, k=1)),4))

print("===================================")
print("Testing different k values for Add-K Smoothing:")
k_values = [0.5, 1, 1.5, 2]  #list of k values to test
for k in k_values:
    #compute bigram perplexity using add-k smoothing with the current k value
    perplexity = compute_bigram_perplexity(val_reviews, lambda w1, w2: add_k_smoothing(w1, w2, k=k))
    print(f"Add-K Perplexity (k={k}): {round(perplexity, 4)}")
