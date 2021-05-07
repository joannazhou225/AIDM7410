#Analysing the text of male of Conservative MP in UK
# 0. installing and loading packages ----- 

# for basic statistics   
library(foreign)
library(psych) 

# for (basic, unsupervised) text mining 
library(NLP) 
library(tm)

# for LDA and topic detection
library(topicmodels)
library(ldatuning)
library(slam)

#check the version of R and what package I have used 
sessionInfo() 

# 1. import data
male_con_01 <- read.csv(file.choose(), header = TRUE, encoding="UTF-8", stringsAsFactors=FALSE)  # file.choose()
#View(female_con_01)
#select variables
names(male_con_01)[names(male_con_01) == 'X.U.FEFF.ID'] <- 'ID'
vars <- c("ID", "Text")

male_con_01_2 <- subset(male_con_01, select = vars) 
names(male_con_01_2) # this is the data element we will going to work on and shared to the class internally

dim(male_con_01_2)
head(male_con_01_2) 

# 2. Inspecting the dataset and variables (with EDAV) 
#look at the variables again
names(male_con_01_2)
#rename some variables
names(male_con_01_2)[names(male_con_01_2) == 'ID'] <- 'doc_id'
names(male_con_01_2)[names(male_con_01_2) == 'Text'] <- 'text'
#check the types of variables
class(male_con_01_2$doc_id) # character 

class(male_con_01_2$text)
#check information 
head(male_con_01_2$text,2)

# 3. unsupervised machine learning for text analytics -----



library(NLP)
library(tm)

## 3.1. Data cleaning for text data  -----

# (data cleaning for  the print media ) 

# to create a VCorpus object  

male_con_docs <- subset(male_con_01_2, select = c("doc_id", "text"))
head(male_con_docs,1)

male_con_VCorpus <- VCorpus(DataframeSource(male_con_docs)) 

# to inspect the contents 
male_con_VCorpus[[1]]$content

#text cleaning
# to convert to all lower cases
male_con_VCorpus <- tm_map(male_con_VCorpus, content_transformer(tolower))  # tm_map: to apply transformation functions (also denoted as mappings) to corpora
male_con_VCorpus[[1]]$content

# to remove URLs 
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
male_con_VCorpus <- tm_map(male_con_VCorpus, content_transformer(removeURL)) 

# to remove anything other than English 
removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x)
male_con_VCorpus <- tm_map(male_con_VCorpus, content_transformer(removeNumPunct)) 

# to remove stopwords, (note: one can define their own myStopwords) 
stopwords("english")
male_con_VCorpus <- tm_map(male_con_VCorpus, removeWords, stopwords("english"))

# to remove extra whitespaces 
male_con_VCorpus <- tm_map(male_con_VCorpus, stripWhitespace) 

# to remove punctuations 
male_con_VCorpus <- tm_map(male_con_VCorpus, removePunctuation)
male_con_VCorpus[[1]]$content



library(SnowballC)
#male_con_VCorpus <- tm_map(female_con_VCorpus, stemDocument)
#male_con_VCorpus[[1]]$content




# 3.2. TF and TF-IDF -----

#for print media
# converting to Document-term matrix (TDM)
male_con_dtm <- DocumentTermMatrix(male_con_VCorpus, control = list(removePunctuation = TRUE, stopwords=TRUE)) 
male_con_dtm

# A high sparsity means terms are not repeated often among different documents.
inspect(male_con_dtm) # a sample of the matrix 

# TF 
term_freq_male <- colSums(as.matrix(male_con_dtm)) 
#save it
write.csv(as.data.frame(sort(rowSums(as.matrix(term_freq_male)), decreasing=TRUE)), file="male_con_dtm_tf.csv")

# TF-IDF 
male_con_dtm_tfidf <- DocumentTermMatrix(male_con_VCorpus, control = list(weighting = weightTfIdf)) # DTM is for TF-IDF calculation 
print(male_con_dtm_tfidf) 
male_con_dtm_tfidf2 = removeSparseTerms(male_con_dtm_tfidf, 0.99)
print(male_con_dtm_tfidf2) 
write.csv(as.data.frame(sort(colSums(as.matrix(male_con_dtm_tfidf2)), decreasing=TRUE)), file="male_con_dtm_tfidf.csv")
inspect(male_con_dtm_tfidf2) 






# 3.3. topic modeling with LDA
library(topicmodels)

# For print media 

# clean the empty (non-zero entry) 

rowTotals_male <- apply(male_con_dtm , 1, sum) #Find the sum of words in each Document
male_con_dtm_nonzero <- male_con_dtm[rowTotals_male> 0, ]

#use metrix to check the number of topic
result <- FindTopicsNumber(
  male_con_dtm_nonzero,
  topics = seq(from = 2, to = 15, by = 1),
  metrics = c("CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 77),
  mc.cores = 2L,
  verbose = TRUE
) # 

FindTopicsNumber_plot(result)
# k7 - 7 topics, 10 term 
male_con_dtm_7topics <- LDA(male_con_dtm_nonzero, k = 7, method = "Gibbs", control = list(iter=2000, seed = 2000)) # find k topics
male_con_dtm_7topics_10words <- terms(male_con_dtm_7topics, 10) # get top 10 words of every topic
(male_con_dtm_7topics_10words <- apply(male_con_dtm_7topics_10words, MARGIN = 2,paste, collapse = ", "))  # show the results immediately, if having a ()  parenthesis 
