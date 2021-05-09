#Analysing the text of female of Conservative MP in UK
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
female_con_01 <- read.csv(file.choose(), header = TRUE, encoding="UTF-8", stringsAsFactors=FALSE)  # file.choose()
#View(female_con_01)
#select variables we need 
names(female_con_01)[names(female_con_01) == 'X.U.FEFF.ID'] <- 'ID'
vars <- c("ID", "Text")

female_con_01_2 <- subset(female_con_01, select = vars) 
names(female_con_01_2) # this is the data element we will going to work on and shared to the class internally

dim(female_con_01_2)
head(female_con_01_2) 

# 2. Inspecting the dataset and variables (with EDAV) 
#look at the variables again
names(female_con_01_2)
#rename some variables
names(female_con_01_2)[names(female_con_01_2) == 'ID'] <- 'doc_id'
names(female_con_01_2)[names(female_con_01_2) == 'Text'] <- 'text'
#check the types of variables
class(female_con_01_2$doc_id) # integer

class(female_con_01_2$text) #character
#check information 
head(female_con_01_2$text,2)

# 3. unsupervised machine learning for text analytics -----




## 3.1. Data cleaning for text data  -----

# to create a VCorpus object  

female_con_docs <- subset(female_con_01_2, select = c("doc_id", "text"))
head(female_con_docs,1)

female_con_VCorpus <- VCorpus(DataframeSource(female_con_docs)) 

# to inspect the contents 
female_con_VCorpus[[1]]$content

#text cleaning
# to convert to all lower cases
female_con_VCorpus <- tm_map(female_con_VCorpus, content_transformer(tolower))  # tm_map: to apply transformation functions (also denoted as mappings) to corpora
female_con_VCorpus[[1]]$content

# to remove URLs 
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
female_con_VCorpus <- tm_map(female_con_VCorpus, content_transformer(removeURL)) 

# to remove anything other than English 
removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x)
female_con_VCorpus <- tm_map(female_con_VCorpus, content_transformer(removeNumPunct)) 

# to remove stopwords, (note: one can define their own myStopwords) 
stopwords("english")
female_con_VCorpus <- tm_map(female_con_VCorpus, removeWords, stopwords("english"))

# to remove extra whitespaces 
female_con_VCorpus <- tm_map(female_con_VCorpus, stripWhitespace) 

# to remove punctuations 
female_con_VCorpus <- tm_map(female_con_VCorpus, removePunctuation)
female_con_VCorpus[[1]]$content



library(SnowballC)
#female_con_VCorpus <- tm_map(female_con_VCorpus, stemDocument)
#female_con_VCorpus[[1]]$content




# 3.2. TF and TF-IDF -----

# converting to Document-term matrix (TDM)
female_con_dtm <- DocumentTermMatrix(female_con_VCorpus, control = list(removePunctuation = TRUE, stopwords=TRUE)) 
female_con_dtm

# A high sparsity means terms are not repeated often among different documents.
inspect(female_con_dtm) # a sample of the matrix 

# TF 
term_freq_female <- colSums(as.matrix(female_con_dtm)) 
#save it
write.csv(as.data.frame(sort(rowSums(as.matrix(term_freq_female)), decreasing=TRUE)), file="female_con_dtm_tf.csv")

# TF-IDF 
female_con_dtm_tfidf <- DocumentTermMatrix(female_con_VCorpus, control = list(weighting = weightTfIdf)) # DTM is for TF-IDF calculation 
print(female_con_dtm_tfidf) 
female_con_dtm_tfidf2 = removeSparseTerms(female_con_dtm_tfidf, 0.99)
print(female_con_dtm_tfidf2) 
write.csv(as.data.frame(sort(colSums(as.matrix(female_con_dtm_tfidf2)), decreasing=TRUE)), file="female_con_dtm_tfidf.csv")
inspect(female_con_dtm_tfidf2) 

#visualization_wordcloud
library("wordcloud2")
#wordcloud_tf
female_tf <- read.csv(file.choose(), header = TRUE, encoding="GBK", stringsAsFactors=FALSE)  # file.choose()
wordcloud2(female_tf,size = 0.6,color = "random-light",shape = "circle",backgroundColor = "gray20")

#wordcloud_tfidf
female_tfidf <- read.csv(file.choose(), header = TRUE, encoding="UTF-8", stringsAsFactors=FALSE)  # file.choose()
wordcloud2(female_tfidf,size = 0.6,color = "random-dark",shape = "star",backgroundColor = "white")



# 3.3. topic modeling with LDA
library(topicmodels)

# For print media 

# clean the empty (non-zero entry) 

rowTotals_female <- apply(female_con_dtm , 1, sum) #Find the sum of words in each Document
female_con_dtm_nonzero <- female_con_dtm[rowTotals_female> 0, ]

#use metrix to check the number of topic
result <- FindTopicsNumber(
  female_con_dtm_nonzero,
  topics = seq(from = 2, to = 15, by = 1),
  metrics = c("CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 77),
  mc.cores = 2L,
  verbose = TRUE
) # 
FindTopicsNumber_plot(result)
# k6 - 6 topics, 10 term 
female_con_dtm_6topics <- LDA(female_con_dtm_nonzero, k = 6, method = "Gibbs", control = list(iter=2000, seed = 2000)) # find k topics
female_con_dtm_6topics_10words <- terms(female_con_dtm_6topics, 10) # get top 10 words of every topic
(female_con_dtm_6topics_10words <- apply(female_con_dtm_6topics_10words, MARGIN = 2,paste, collapse = ", "))  # show the results immediately, if having a ()  parenthesis 

