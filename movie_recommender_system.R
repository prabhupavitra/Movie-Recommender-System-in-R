##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)

edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

################################################
# Helper Functions for EDA and Model Selection
################################################

# This function creates one-hot encoded columns of a selected column
one_hot_encoding <- function(df,colname,delim){
  conv_list <- df[[colname]] %>% 
    paste(sep="",collapse=delim) 
  
  # Get the list of possible values category can have
  possible_val <- unique(strsplit(conv_list,"\\|")[[1]])
  temp_df <- cbind(df,sapply(possible_val, function(g) {
    as.numeric(str_detect(df[[colname]], g))
  })
  )
  colnames(temp_df) <- c(names(df), possible_val)
  return(temp_df)
}

### Loss functions 
# This function calculates the Root Mean Squared Error (RMSE) 
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# This function calculates the Mean Absolute Error (MAE)
MAE <- function(true_ratings, predicted_ratings){
  mean(abs(true_ratings - predicted_ratings))
}

##################################################
# Exploratory Data Analysis on the movielens data
##################################################

# Structure of edx dataset
str(edx,vec.len=2.5) %>%
  print.data.frame() 

# dataset dimensions
dim(edx)

# Counts the number of NAs in each column of the dataset
colSums(is.na(edx))

# Variables available in edx dataset
vars <- names(edx)

# Viewing the contents of dataset
head(edx)

#### Data preparation for exploratory data analysis(EDA)

# Creates a dataset with encoded genres column of edx
encoded_edx <- one_hot_encoding(edx,"genres","|")

# First 6 rows of encoded edx dataset
tibble(head(encoded_edx)) %>% 
  dplyr::select(-all_of(vars[1:5])) %>% 
  select("genres","Comedy","Romance","Action","Crime","Thriller", "Drama", "Sci-Fi","Adventure")

####################################################
# Data Visualizations for Exploratory Data Analysis
####################################################

# Frequency distribution of user ratings
edx %>%
  group_by(rating) %>%
  summarise(num_rating = n()) %>%
  ggplot(aes(rating,num_rating/1000)) +
  geom_bar(stat="identity") +
  xlab("Rating") +
  ylab("Number of ratings (in thousands)") +
  theme(plot.title = element_text(hjust = 0.5,face="italic"))

# Number of unique movies in edx dataset
length(unique(edx$movieId))


# Density plot of distribution of ratings grouped by movieId
edx %>% 
  group_by(movieId) %>%
  summarise(num_rating = n())  %>%
  ggplot(aes(num_rating)) +
  geom_density() +
  ylab("Density") +
  theme(plot.title = element_text(hjust = 0.5,face="italic"),
        axis.title.x = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank())


# Normalized ecdf of the number of ratings per movie
edx %>% 
  group_by(movieId) %>%
  summarise(num_rating = n()) %>% 
  ggplot(aes(x = num_rating)) + 
  stat_ecdf(geom = "step", 
            size = 0.75, 
            pad = FALSE) +
  theme_minimal() + 
  coord_cartesian(clip = "off") +
  geom_vline(xintercept = 1000, colour = "darkgrey",alpha=0.5,linetype="dashed") +
  geom_text(aes(1000, 0, label=1000,vjust=1.5),
            size=3,family="Courier", fontface="italic",color="darkgrey") +
  geom_hline(yintercept = 0.80, colour = "darkgrey",alpha=0.5,linetype="dashed") +
  geom_text(aes(0, 0.8, label=0.8,hjust=-1),
            size=3,family="Courier", fontface="italic",color="darkgrey") +
  geom_hline(yintercept = 0.99, colour = "darkgrey",alpha=0.5,linetype="dashed") +
  geom_text(aes(0, 0.99, label=0.99,hjust=-1),
            size=3,family="Courier", fontface="italic",color="darkgrey") +  
  scale_x_continuous(limits = c(0,35000), 
                     expand = c(0, 0),
                     name="Number of ratings") +
  scale_y_continuous(limits = c(0, 1.01), 
                     expand = c(0, 0), 
                     name = "Cumulative Frequency") +
  theme(plot.title=element_text(hjust = 0.5,face="italic"),
        axis.text.x = element_text(angle=0))        

# Number of unique users in edx dataset
length(unique(edx$userId))

# Density plot of distribution of ratings grouped by userId
edx %>%
  group_by(userId) %>% 
  summarize(num_rating=n()) %>% 
  arrange(num_rating) %>%
  ggplot(aes(num_rating)) +
  geom_density() +
  ylab("Density") +
  theme(plot.title = element_text(hjust = 0.5,face="italic"),
        axis.title.x = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank())

# Normalized ecdf of the number of ratings per user
edx %>% 
  group_by(userId) %>%
  summarise(num_rating = n()) %>% 
  ggplot(aes(x = num_rating)) + 
  stat_ecdf(geom = "step", 
            size = 0.75, 
            pad = FALSE) +
  theme_minimal() + 
  coord_cartesian(clip = "off") +
  geom_vline(xintercept = 270, colour = "darkgrey",alpha=0.5,linetype="dashed") +
  geom_text(aes(270, 0, label=270,vjust=1.5),
            size=3,color="darkgrey",family="Courier", fontface="italic") +
  geom_hline(yintercept = 0.90, colour = "darkgrey",alpha=0.5,linetype="dashed") +
  geom_text(aes(0, 0.9, label=0.9,hjust=1.5),
            size=3,color="darkgrey",family="Courier", fontface="italic") +
  scale_x_continuous(limits = c(0,7000), 
                     expand = c(0, 0),
                     name="Number of ratings") +
  scale_y_continuous(limits = c(0, 1.01), 
                     expand = c(0, 0), 
                     name = "Cumulative Frequency") +
  theme(plot.title=element_text(hjust = 0.5,face="italic"),
        axis.text.x = element_text(angle=0))

# Overview of Year data in edx
edx %>% 
  mutate(date = date(as_datetime(timestamp, origin="1970-01-01"))) %>%
  mutate(year = year(date)) %>%
  pull(year) %>% unique() %>% sort()


# Average rating by day
edx %>% 
  mutate(Day = day(date(as_datetime(timestamp, origin="1970-01-01")))) %>%
  group_by(Day) %>%
  summarise(avg_rating=mean(rating)) %>%
  ggplot(aes(Day, avg_rating)) +
  geom_point() +
  geom_smooth() +
  xlab("Day of the month") +
  ylab("Average rating") +
  theme(plot.title=element_text(hjust = 0.5,face="italic"))


# Average rating of movies of each genre
data.frame(avg_rating = 
             sapply(names(dplyr::select(encoded_edx,-all_of(names(edx)))),
                    function(i){
                      mean(encoded_edx$rating[as.logical(encoded_edx[[i]])])})) %>%
  mutate(genre = rownames(.)) %>%
  ggplot(aes(reorder(genre,avg_rating),avg_rating)) +
  geom_bar(stat="identity") +
  coord_flip() +
  xlab("") +
  ylab("Average Rating") +
  theme(plot.title=element_text(hjust = 0.5,face="italic"))

# Number of movies of each genre
data.frame(num_movies = encoded_edx %>%
             dplyr::select(-all_of(vars),"movieId") %>%
             distinct() %>%
             colSums()) %>%
  mutate(genre = rownames(.)) %>%
  filter(genre!="movieId") %>%
  ggplot(aes(reorder(genre,num_movies),num_movies)) +
  geom_point() +
  coord_flip() +
  xlab("") +
  ylab("Number of movies") +
  theme(plot.title=element_text(hjust = 0.5,face="italic"))

# Number of ratings of each genre
data.frame(num_ratings = encoded_edx %>%
             dplyr::select(-all_of(vars)) %>%
             colSums()) %>%
  mutate(genre = rownames(.)) %>%
  ggplot(aes(reorder(genre,num_ratings),num_ratings/1000000)) +
  geom_bar(stat="identity") +
  coord_flip() +
  xlab("") +
  ylab("Number of Ratings \n(in million)") +
  theme(plot.title=element_text(hjust = 0.5,face="italic"))

# Release Year Patterns
edx %>%
  mutate(ReleaseYear  = as.numeric(sub("\\).*", "", 
                                       sub(".*\\(", "", str_sub(title,-6,-1))))) %>%
  group_by(ReleaseYear) %>%
  summarise(avg_rating=mean(rating)) %>%
  ggplot(aes(ReleaseYear, avg_rating)) +
  geom_point() +
  geom_smooth() +
  xlab("Release Year of movies") +
  ylab("Average rating") +
  theme(plot.title=element_text(hjust = 0.5,face="italic"))

################################################
# Handling Missing data and unstable data points
################################################
# Identifies rows to be removed
edx %>% filter(genres=="(no genres listed)") %>% print.data.frame()

# Removes the unstable datapoints in edx dataset as identified above
edx <- edx %>% filter(genres!="(no genres listed)")
# Removes from one-hot encoded form of edx dataset
encoded_edx <- encoded_edx %>% 
  dplyr::select(-c("(no genres listed)")) %>%
  filter(genres!="(no genres listed)")

# before deletion : 9000055; 
# after deletion  : 9000048
dim(edx)
dim(encoded_edx)

##########################################################
# Create train set, test set
##########################################################
## Create train and test sets for cross validation
# Create train set and test set
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = edx$rating, times = 1 ,p=0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in validation set are also in edx set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)

#########################################
#  Data Preparation : Train and Test set
#########################################

# Extracting Day from timestamp column and Release year of the movie from title column : train_set
train_set <- train_set %>% 
  mutate(Day = day(as_datetime(train_set[["timestamp"]]))) %>% 
  mutate(ReleaseYear  = as.numeric(sub("\\).*", "", sub(".*\\(", "", str_sub(title,-6,-1)))))

# Extracting Day from timestamp column and Release year of the movie from title column : test_set
test_set <- test_set %>% 
  mutate(Day = day(as_datetime(test_set[["timestamp"]]))) %>%
  mutate(ReleaseYear  = as.numeric(sub("\\).*", "", sub(".*\\(", "", str_sub(title,-6,-1))))) 

# One-hot encoding on the genres column
train_set <- one_hot_encoding(train_set,"genres","|")
test_set <- one_hot_encoding(test_set,"genres","|")

############################################################
# Machine Learning Models for the 10M movielens dataset
############################################################


################   Model 1 : Linear Regression   #############

# Model 1a : Initial Predictions of average rating 
# Average rating as the predicted rating
mu <- mean(train_set$rating)

# Predictions with test set for the first model
naive_rmse <- RMSE(test_set$rating, mu)
naive_mae <- MAE(test_set$rating, mu)


# Calculating rmse and mae for the model with average rating
rmse_results <- tibble(method = "Average rating", 
                       RMSE = naive_rmse,
                       MAE = naive_mae)
# Display results
rmse_results %>% knitr::kable()
# Deleting used objects and variables 
rm(naive_rmse,naive_mae)


# Model 1b : Including Movie Effects to Initial Predictions

# Estimation of Movie bias
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Predictions on test set with estimated movie bias
preds_movie <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

# Calculating RMSE for the model with Movie effects
rmse_movie <- RMSE(test_set$rating,preds_movie)
# Calculating MAE for the model with Movie effects
mae_movie <- MAE(test_set$rating,preds_movie)

# Results
rmse_results <- 
  rbind(rmse_results,tibble(method = "Movie Effects", 
                            RMSE = rmse_movie,
                            MAE = mae_movie))
rmse_results %>% knitr::kable()
rm(preds_movie,rmse_movie,mae_movie)

#  Model 1c : Including User Effects and Movie Effects to Initial Predictions

# Estimation of User bias
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Predictions on test set with estimated movie and user bias
preds_movie_user <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Calculating rmse and mae for the model with Movie and User effects 
rmse_movie_user <- RMSE(test_set$rating,preds_movie_user)
mae_movie_user <- MAE(test_set$rating,preds_movie_user)

# Results
rmse_results <- 
  rbind(rmse_results,tibble(method = "Movie + User Effects", 
                            RMSE = rmse_movie_user,
                            MAE = mae_movie_user))
rmse_results %>% knitr::kable()

# Deleting used objects and variables 
rm(preds_movie,rmse_movie_user,mae_movie_user)

# Model 1d: Including day effects with movie and user effects to Initial Predictions

# Estimation of day effects
day_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(Day) %>%
  summarize(b_t = mean(rating - mu - b_i - b_u)) 

# Smoothing the values of b_t
f_smooth <- loess(b_t ~ Day,data = day_avgs,family="gaussian",span=1)

# Predictions on test set with estimated movie, user and day bias
preds_movie_user_day <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u  + f_smooth$fitted[Day]) %>%
  pull(pred)

# Calculating rmse and mae for the model with Movie, User and Day effects 
rmse_movie_user_day <- RMSE(test_set$rating,preds_movie_user_day)
mae_movie_user_day <- MAE(test_set$rating,preds_movie_user_day)

# Results
rmse_results <- 
  rbind(rmse_results,tibble(method = "Movie + User + Day Effects", 
                            RMSE = rmse_movie_user_day,
                            MAE = mae_movie_user_day))
rmse_results %>% knitr::kable()

# Deleting used objects and variables 
rm(preds_movie_user_day,rmse_movie_user_day,mae_movie_user_day)


# Model 1e : Including release year effects with movie, user effects to Initial Predictions

# Estimation of release year effects
relyear_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(ReleaseYear) %>%
  summarize(b_ry = mean(rating - mu - b_i - b_u)) 

# Predictions on test set with estimated movie, user, day and release year bias
preds_movie_user_relyear <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(relyear_avgs, by='ReleaseYear') %>%
  mutate(pred = mu + b_i + b_u + b_ry) %>%
  pull(pred)

# Calculating rmse and mae for the model with Movie, User and Day effects 
rmse_movie_user_relyear <- RMSE(test_set$rating,preds_movie_user_relyear)
mae_movie_user_relyear <- MAE(test_set$rating,preds_movie_user_relyear)

# Results
rmse_results <- 
  rbind(rmse_results,tibble(method = "Movie + User + Release Year Effects", 
                            RMSE = rmse_movie_user_relyear,
                            MAE = mae_movie_user_relyear ))
rmse_results %>% knitr::kable()

# Deleting used objects and variables 
rm(preds_movie_user_relyear,rmse_movie_user_relyear,mae_movie_user_relyear)


# Model 1f : Including Genre effects with movie, user , day and release year effects to Initial Predictions
# Installing required package for ginv()
if(!require(MASS)) install.packages("MASS", repos = "http://cran.us.r-project.org")
# Note that MASS library has a select function; Use dplyr::select for using select function from dplyr
library(MASS)
```
```{r genre_effects, echo=TRUE}
# Estimating bias due to genre and genre columns
genre_df <- 
  train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(relyear_avgs, by='ReleaseYear') %>%
  mutate(g_ui = rating - mu - b_i - b_u - b_ry) %>%
  dplyr::select(-all_of(vars),-any_of(c("b_i","b_u","b_ry","Day","ReleaseYear")),"g_ui")

# Converting training data to matrix form
x <- as.matrix(genre_df[,-c("g_ui")])
y <- as.matrix(genre_df[,c("g_ui")])

# Estimating co-efficient of individual genres using Moore-Penrose pseudoinverse
beta_df <- data.frame(
  beta =  ginv(x) %*% y,
  genre = names(genre_df[,-c("g_ui")]))

# Estimating predictions for genre effects
g_ui_hat <- rowSums(sapply(beta_df$genre,
                           function(i){
                             as.numeric(beta_df[,1][beta_df$genre==i]) * test_set[[i]]
                           }))

# Plotting coefficients for all genres
beta_df %>% 
  setnames(c("beta","genre")) %>%
  ggplot(aes(beta,genre)) + 
  geom_point() + 
  geom_vline(xintercept=0, linetype=2) +
  xlab("Genre") +
  ylab("Beta") +
  theme(plot.title=element_text(hjust = 0.5,face="italic"))

# Estimating rating predictions
preds_movie_user_day_genre <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(relyear_avgs, by='ReleaseYear') %>%
  cbind(g_ui_hat) %>% 
  mutate(pred = mu + b_i + b_u + g_ui_hat + b_ry ) %>%
  pull(pred)

# Calculating rmse and mae for the model with Movie, User and genre effects 
rmse_genre <- RMSE(test_set$rating,preds_movie_user_day_genre)
mae_genre <- MAE(test_set$rating,preds_movie_user_day_genre)
  
# Binding genre RMSE to results
rmse_results <- 
  rbind(rmse_results,tibble(method = "Movie + User + Release Year + Genre Effects", 
                            RMSE = rmse_genre,
                            MAE = mae_genre ))
rmse_results %>% knitr::kable()

# Deleting used objects and variables 
rm(genre_df,x,y,beta_df,preds_movie_user_day_genre,rmse_genre,mae_genre,g_ui_hat,
   mu,movie_avgs,user_avgs,day_avgs,f_smooth,relyear_avgs)


################   Model 2 :  Linear Model with Regularization of movieId and userId   #############


# Optimize lambda by minimizing RMSE
lambdas <- seq(1, 10, 0.25)
mu <- mean(train_set$rating) 

rmses <- sapply(lambdas, function(l){
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n()+l))
  
  preds_reglrzed_movie_user <- test_set %>% 
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    mutate(pred = mu + b_i + b_u ) %>%
    pull(pred)
  
  return(RMSE(test_set$rating,preds_reglrzed_movie_user))
})

# Visualizing RMSE over a sequence of lambda values 
qplot(lambdas, rmses)  +
  theme(plot.title = element_text(hjust = 0.5,face="italic"))

# Optimal lambda for minimizing rmse
lambda <- lambdas[which.min(rmses)]

# Deleting used objects and variables 
rm(lambdas,rmses)

### Regularized model with optimal lambda

b_i <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

b_u <- train_set %>%
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+lambda))

preds_reglrzed_movie_user <- test_set %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

rmse_reglrzed_movie_user <- RMSE(test_set$rating,preds_reglrzed_movie_user)
mae_reglrzed_movie_user <- MAE(test_set$rating,preds_reglrzed_movie_user)

# Results
rmse_results <- 
  rbind(rmse_results,tibble(method = "Regularized Movie + User", 
                            RMSE = rmse_reglrzed_movie_user,
                            MAE = mae_reglrzed_movie_user))
rmse_results %>% knitr::kable()

# Deleting used objects and variables 
rm(preds_reglrzed_movie_user,rmse_reglrzed_movie_user,mae_reglrzed_movie_user)

## Including Genre effects and Release year effects to the Regularized movie and user model

# Estimating release year bias
b_ry <- train_set %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  group_by(ReleaseYear) %>%
  summarize(b_ry = mean(rating - mu - b_i - b_u )) 
  
# Estimating bias due to genre and genre columns
genre_df_reg <- 
  train_set %>%
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_ry, by='ReleaseYear') %>%
  mutate(g_ui = rating - mu - b_i - b_u - b_ry ) %>%
  dplyr::select(-all_of(vars),-any_of(c("b_i","b_u","Day","b_ry","ReleaseYear")),"g_ui")

# Converting training data to matrix form
x <- as.matrix(genre_df_reg[,-c("g_ui")])
y <- as.matrix(genre_df_reg[,c("g_ui")])

# Estimating co-efficient of individual genres using Moore-Penrose pseudoinverse
beta_df_reg <- data.frame(
  beta =  ginv(x) %*% y,
  genre = names(genre_df_reg[,-c("g_ui")]))

# Visualizing coefficients of genre with regularized model
beta_df_reg %>% 
  setnames(c("beta","genre")) %>%
  ggplot(aes(beta,genre)) + 
  geom_point() + 
  geom_vline(xintercept=0, linetype=2) +
  xlab("Genre") +
  ylab("Beta") +
  theme(plot.title=element_text(hjust = 0.5,face="italic"))

# Estimating predictions for genre effects
g_ui_hat_reg <- rowSums(sapply(beta_df_reg$genre,
                               function(i){
                                 as.numeric(beta_df_reg[,1][beta_df_reg$genre==i]) * test_set[[i]]
                               }))

# Estimating rating predictions
preds_regularized_iu_genre <- test_set %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_ry, by='ReleaseYear') %>%
  cbind(g_ui_hat_reg) %>% 
  mutate(pred = mu + b_i + b_u + b_ry + g_ui_hat_reg) %>%
  pull(pred)

# Calculating rmse and mae for the model with Movie, User and genre effects 
rmse_genre_reg <- RMSE(test_set$rating,preds_regularized_iu_genre)
mae_genre_reg <- MAE(test_set$rating,preds_regularized_iu_genre)

# RMSE using co-efficients estimated with Moore-Penrose pseudoinverse
rmse_results <- 
  rbind(rmse_results,tibble(method = "Regularized Movie & User + Genre + Release Year Effects Model", 
                            RMSE = rmse_genre_reg,
                            MAE = mae_genre_reg ))
rmse_results %>% knitr::kable()

# Deleting used objects and variables 
rm(genre_df_reg,beta_df_reg,preds_regularized_iu_genre,
   rmse_genre_reg,mae_genre_reg,g_ui_hat_reg,x,y,lambda,b_i,b_u,mu,b_ry)


################   Model 3 :  Matrix Factorization using recosystem package   #############

# Installing recosystem pkg
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
# Loading recosystem library for recommender object
library(recosystem)

## Collaborative filtering using Matrix Factorization
train_vector <- data_memory(user_index = train_set$userId,
                            item_index = train_set$movieId,
                            rating = train_set$rating,
                            index1=T)

test_vector <- data_memory(user_index = test_set$userId,
                           item_index = test_set$movieId,
                           rating = test_set$rating,
                           index1=T)
# Constructing a Recommender System Object
recommender = recosystem::Reco()

#  Uses cross validation to tune the model parameters
opts <- recommender$tune(train_vector, 
                         opts = list(dim = c(10, 20, 30),  
                                     lrate = c(0.1, 0.2), # learning rate
                                     costp_l1 = 0,  # L1 regularization cost for user factors
                                     costq_l1 = 0, # L1 regularization cost for item factors
                                     costp_l2 = c(0.01,0.1), # L2 regularization cost for user factors
                                     costq_l2 = c(0.01,0.1), # L2 regularization cost for item factors
                                     nthread = 4,  # number of threads for Parallel computation
                                     niter = 10)) # whether to show detailed information. 

# Trains a recommender model with the training data source
recommender$train(train_vector,opts = c(opts$min, 
                                        nthread = 4, 
                                        niter = 300, 
                                        verbose = FALSE))

# Predicts unknown entries in the rating matrix
preds_matrix_fact <- recommender$predict(test_vector, out_memory())

# Estimating rmse and mae for the matrix factorization model
rmse_matrix_fact <- RMSE(test_set$rating, preds_matrix_fact)
mae_matrix_fact <- MAE(test_set$rating, preds_matrix_fact)

# results
rmse_results <- 
  rbind(rmse_results,tibble(method = "Matrix Factorization (recosystem package)", 
                            RMSE = rmse_matrix_fact,
                            MAE = mae_matrix_fact ))
rmse_results %>% knitr::kable()

# Deleting used objects and variables 
rm(train_vector,test_vector,preds_matrix_fact,rmse_matrix_fact,mae_matrix_fact)

#######################################################################
# RMSE of finalized model(Matrix Factorization) with the validation set 
#######################################################################
# Set random seed 
set.seed(1234, sample.kind = "Rounding")

# Convert 'edx' and 'validation' sets to recosystem input format
edx_vector <- data_memory(user_index = edx$userId,
                          item_index = edx$movieId,
                          rating = edx$rating,
                          index1=T)

validation_vector <- data_memory(user_index = validation$userId,
                                 item_index = validation$movieId,
                                 rating = validation$rating,
                                 index1=T)
# Create the model object
recommender <- Reco()

# Train the model
recommender$train(edx_vector,opts = c(opts$min, 
                                      nthread = 4, 
                                      niter = 300, 
                                      verbose = FALSE))

# Estimating predictions using recommender
preds_validation <- recommender$predict(validation_vector, out_memory())
# RMSE for matrix factorization
rmse_validation <- RMSE(validation$rating, preds_validation)
mae_validation <- MAE(validation$rating, preds_validation)

# Tabulating the validation results
rmse_validation_results <- tibble("Final Model" = "Matrix Factorization", 
                                  "RMSE on Validation Set" = rmse_validation,
                                  "MAE on Validation Set" = mae_validation)
rmse_validation_results %>% knitr::kable()

# Deleting used objects and variables 
rm(edx_vector,validation_vector,preds_validation,rmse_validation,mae_validation)
# rm(rmse_results,execution_time,rmse_validation_results,train_set,test_set,edx,validation)

