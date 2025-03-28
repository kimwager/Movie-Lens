###############################################################
# MovieLens Recommendation System
# 
# This script builds and evaluates a movie recommendation system
# using the MovieLens 10M dataset. It implements progressively 
# more sophisticated models to predict user ratings.
#
# Dataset: MovieLens 10M (https://grouplens.org/datasets/movielens/10m/)
# Author: Kim Wager
# Date: 07/03/25
###############################################################

###############################################################
# PART 1: DATA PREPARATION
###############################################################

##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this code was provided as part of the course assessment

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

# Read and format the ratings data from the DAT file
ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

# Read and format the movies data from the DAT file
movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                       stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

# Join ratings and movies data to create the complete dataset
movielens <- left_join(ratings, movies, by = "movieId")

# Create train/test split for model evaluation
# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
# This ensures we don't test on users or movies we haven't seen during training
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

###############################################################
# PART 2: MODEL DEVELOPMENT STRATEGY
#
# We use a progressive approach to build our recommendation system:
# 1. Baseline model: Global mean rating
# 2. Movie effects model: Accounts for movie popularity
# 3. User effects model: Accounts for user rating tendencies
# 4. Combined model: Incorporates both movie and user effects
# 5. Regularized model: Prevents overfitting with regularization
# 6. Genre effects: Adds genre-based preferences
#
# For each model, we:
#   - Train on the edx_train dataset
#   - Evaluate on the validation dataset
#   - Compare performance using RMSE (Root Mean Square Error)
###############################################################

# Split data for model development
# - edx_train (80% of edx): Used to train the models
# - validation (20% of edx): Used to tune and evaluate model performance
# - final_holdout_test: Completely separate dataset used only for final evaluation

set.seed(1, sample.kind="Rounding")
validation_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE) # uses createDataPartition() with p = 0.2 to select 20% of the rows from edx
validation <- edx[validation_index,]
edx_train <- edx[-validation_index,]

###############################################################
# PART 3: BASELINE MODEL
# 
# The simplest possible model: predict the same rating (overall mean)
# for every movie and user. This establishes a performance floor
# that more complex models should improve upon.
###############################################################

## Step 1: Calculate the mean rating (μ)

# Calculate the overall mean rating from training data
mu <- mean(edx_train$rating)
print(paste("Overall mean rating:", round(mu, 4)))


## Step 2: Create predictions for validation set

# Create predictions using just the mean (mu) from the previous step
predictions <- rep(mu, nrow(validation))


## Step 3: Define RMSE function and calculate error

# Define RMSE (Root Mean Square Error) function
# RMSE measures the average magnitude of prediction errors
# Lower RMSE values indicate better prediction accuracy
# 
# Parameters:
#   true_ratings: Vector of actual ratings from users
#   predicted_ratings: Vector of model-predicted ratings
# 
# Returns:
#   Numeric value representing prediction error (lower is better)
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Calculate RMSE for baseline model
# Print the mean rating with 4 decimal places for clarity
naive_rmse <- RMSE(validation$rating, predictions)
print(paste("RMSE for baseline model:", round(naive_rmse, 4)))


## Step 4: Create results table to track model performance

# Create results table
rmse_results <- tibble(
  Method = "Baseline model",
  RMSE = naive_rmse
)

knitr::kable(rmse_results, digits = 4, 
             caption = "Baseline model performance")


###############################################################
# PART 4: MOVIE EFFECTS MODEL
# 
# This model accounts for the fact that some movies are generally
# rated higher or lower than others, regardless of who rates them.
# 
# For each movie i:
#   b_i = average(rating - μ)
# 
# Where:
#   μ is the global average rating
#   b_i is the "movie effect" or movie bias
# 
# Prediction formula:
#   predicted_rating = μ + b_i
###############################################################

# Calculate global mean (as previously)
mu <- mean(edx_train$rating)

# Calculate movie effect (b_i)
# Group by movie and calculate the average deviation from the mean rating
movie_avgs <- edx_train %>%
   group_by(movieId) %>%
   summarize(b_i = mean(rating - mu))

# Predict ratings using global mean + movie effect
predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  mutate(
    # Replace NA movie effects with 0 (global mean), in case some movies are in the validation set but not the training set
    b_i = ifelse(is.na(b_i), 0, b_i),
    pred = mu + b_i
  ) %>% 
  pull(pred)

# Calculate Root Mean Square Error (RMSE)
movie_effect_rmse <- RMSE(validation$rating, predicted_ratings)

# Print results
print(paste("Movie Effects RMSE:", movie_effect_rmse))

# Calculate improvement over naive RMSE
improvement <- naive_rmse - movie_effect_rmse
improvement_percentage_movie_effects <- (naive_rmse - movie_effect_rmse) / naive_rmse * 100

# Add to results table
rmse_results <- bind_rows(rmse_results,
                         tibble(Method = "Movie Effects Model",
                                RMSE = movie_effect_rmse))


###############################################################
# PART 5: USER EFFECTS MODEL
# 
# This model accounts for the fact that some users tend to rate
# movies higher or lower than the average user.
# 
# For each user u:
#   b_u = average(rating - μ)
# 
# Prediction formula:
#   predicted_rating = μ + b_u
###############################################################

# Calculate user-specific effects (b_u)
user_avgs <- edx_train %>%
   group_by(userId) %>%
   summarize(b_u = mean(rating - mu))

# Predict ratings using global mean + user effect
predicted_ratings <- validation %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(
    # Handle missing users
    b_u = ifelse(is.na(b_u), 0, b_u),
    pred = mu + b_u
  ) %>% 
  pull(pred)

# Calculate RMSE
user_effect_rmse <- RMSE(validation$rating, predicted_ratings)

print(paste("User Effects RMSE:", user_effect_rmse))

# Calculate improvement over naive RMSE
improvement <- naive_rmse - user_effect_rmse
improvement_percentage_user_effects <- (naive_rmse - user_effect_rmse) / naive_rmse * 100

# Add to results
rmse_results <- bind_rows(rmse_results,
                         tibble(Method = "User Effects Model",
                                RMSE = user_effect_rmse))

# Display results table
knitr::kable(rmse_results, digits = 4,
             caption = "Model performance on validation set")                              


###############################################################
# PART 6: COMBINED MOVIE AND USER EFFECTS MODEL
# 
# This model incorporates both movie and user biases for more
# accurate predictions.
# 
# Model components:
#   μ: Global mean rating
#   b_i: Movie effect for movie i
#   b_u: User effect for user u
# 
# Prediction formula:
#   predicted_rating = μ + b_i + b_u
###############################################################

# Calculate global mean (as previously)
mu <- mean(edx_train$rating)

# Calculate movie effects (as previously)
movie_avgs <- edx_train %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# Calculate user effects accounting for movie effects
# This ensures we don't double-count effects
user_avgs <- edx_train %>%
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))


# Make predictions on validation set
predicted_ratings <- validation %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(
    b_i = ifelse(is.na(b_i), 0, b_i),
    b_u = ifelse(is.na(b_u), 0, b_u),
    pred = mu + b_i + b_u
  ) %>%
  pull(pred)

# Calculate RMSE
combined_effect_rmse <- RMSE(validation$rating, predicted_ratings)
print(paste("Combined Effects RMSE (validation):", combined_effect_rmse))

# Calculate improvement over naive RMSE
improvement <- naive_rmse - combined_effect_rmse
improvement_percentage_combined_effects <- (naive_rmse - combined_effect_rmse) / naive_rmse * 100

# Add to results
rmse_results <- bind_rows(rmse_results,
                         tibble(Method = "Combined Effects Model",
                                RMSE = combined_effect_rmse))

# Display results table
knitr::kable(rmse_results, digits = 4,
             caption = "Model performance on validation set")

###############################################################
# PART 7: REGULARIZED MODEL
# 
# Regularization penalizes extreme estimates that may be based on
# small sample sizes. This helps prevent overfitting.
# 
# For movie effects:
#   b_i = sum(rating - μ)/(n_i + λ)
# 
# For user effects:
#   b_u = sum(rating - μ - b_i)/(n_u + λ)
# 
# Where:
#   n_i = number of ratings for movie i
#   n_u = number of ratings by user u
#   λ = regularization parameter
#
# We test different λ values to find the optimal balance between
# fitting the training data and generalizing to new data.
###############################################################

# Create a container list for all model components
movie_model <- list()

# Test a range of lambda values for regularization
movie_model$lambdas <- seq(0, 10, 0.25)
movie_model$rmses <- sapply(movie_model$lambdas, function(l){
  
  # Regularized movie effects
  movie_reg_avgs <- edx_train %>%
    group_by(movieId) %>%
    summarize(
      b_i = sum(rating - mu)/(n() + l),
      n_i = n()
    )
  
  # Regularized user effects
  user_reg_avgs <- edx_train %>%
    left_join(movie_reg_avgs, by = "movieId") %>%
    group_by(userId) %>%
    summarize(
      b_u = sum(rating - mu - b_i)/(n() + l),
      n_u = n()
    )
  
  # Make predictions
  predicted_ratings <- validation %>%
    left_join(movie_reg_avgs, by = "movieId") %>%
    left_join(user_reg_avgs, by = "userId") %>%
    mutate(
      b_i = ifelse(is.na(b_i), 0, b_i),
      b_u = ifelse(is.na(b_u), 0, b_u),
      pred = mu + b_i + b_u
    ) %>%
    pull(pred)
  
  return(RMSE(validation$rating, predicted_ratings))
})

# Find optimal lambda and save model components
movie_model$optimal_lambda <- movie_model$lambdas[which.min(movie_model$rmses)]
print(paste("Optimal lambda:", movie_model$optimal_lambda))

# Save final effects using optimal lambda
movie_model$movie_effects <- edx_train %>%
  group_by(movieId) %>%
  summarize(
    b_i = sum(rating - mu)/(n() + movie_model$optimal_lambda),
    n_i = n()
  )

movie_model$user_effects <- edx_train %>%
  left_join(movie_model$movie_effects, by = "movieId") %>%
  group_by(userId) %>%
  summarize(
    b_u = sum(rating - mu - b_i)/(n() + movie_model$optimal_lambda),
    n_u = n()
  )

# Plot RMSE vs lambda to visualize the impact of regularization
qplot(movie_model$lambdas, movie_model$rmses) +
  geom_line() +
  xlab("Lambda") +
  ylab("RMSE") +
  ggtitle("RMSE vs Regularization Parameter")

# Add results to the comparison table
rmse_results <- bind_rows(rmse_results,
                         tibble(Method = "Regularized Model",
                                RMSE = min(movie_model$rmses)))

# Display updated results
knitr::kable(rmse_results, digits = 4,
             caption = "Model performance on validation set")

###############################################################
# PART 8: GENRE EFFECTS MODEL WITH OPTIMIZED REGULARIZATION
###############################################################

# Test a range of lambda values for all effects
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  
  # Regularized movie effects
  movie_reg_avgs <- edx_train %>%
    group_by(movieId) %>%
    summarize(
      b_i = sum(rating - mu)/(n() + l),
      n_i = n()
    )
  
  # Regularized user effects
  user_reg_avgs <- edx_train %>%
    left_join(movie_reg_avgs, by = "movieId") %>%
    group_by(userId) %>%
    summarize(
      b_u = sum(rating - mu - b_i)/(n() + l),
      n_u = n()
    )
  
  # Regularized genre effects
  genre_reg_avgs <- edx_train %>%
    left_join(movie_reg_avgs, by = "movieId") %>%
    left_join(user_reg_avgs, by = "userId") %>%
    group_by(genres) %>%
    summarize(
      b_g = sum(rating - mu - b_i - b_u)/(n() + l),
      n_g = n()
    )
  
  # Make predictions
  predicted_ratings <- validation %>%
    left_join(movie_reg_avgs, by = "movieId") %>%
    left_join(user_reg_avgs, by = "userId") %>%
    left_join(genre_reg_avgs, by = "genres") %>%
    mutate(
      b_i = ifelse(is.na(b_i), 0, b_i),
      b_u = ifelse(is.na(b_u), 0, b_u),
      b_g = ifelse(is.na(b_g), 0, b_g),
      pred = mu + b_i + b_u + b_g
    ) %>%
    pull(pred)
  
  return(RMSE(validation$rating, predicted_ratings))
})

# Find optimal lambda and save model components
optimal_lambda <- lambdas[which.min(rmses)]
print(paste("Optimal lambda:", optimal_lambda))

# Save final effects using optimal lambda
final_movie_effects <- edx_train %>%
  group_by(movieId) %>%
  summarize(
    b_i = sum(rating - mu)/(n() + optimal_lambda),
    n_i = n()
  )

final_user_effects <- edx_train %>%
  left_join(final_movie_effects, by = "movieId") %>%
  group_by(userId) %>%
  summarize(
    b_u = sum(rating - mu - b_i)/(n() + optimal_lambda),
    n_u = n()
  )

final_genre_effects <- edx_train %>%
  left_join(final_movie_effects, by = "movieId") %>%
  left_join(final_user_effects, by = "userId") %>%
  group_by(genres) %>%
  summarize(
    b_g = sum(rating - mu - b_i - b_u)/(n() + optimal_lambda),
    n_g = n()
  )

# Plot RMSE vs lambda
qplot(lambdas, rmses) +
  geom_line() +
  xlab("Lambda") +
  ylab("RMSE") +
  ggtitle("RMSE vs Regularization Parameter (All Effects)")

# Make final predictions including genre effects
predicted_ratings_with_genre <- validation %>%
  left_join(final_movie_effects, by = "movieId") %>%
  left_join(final_user_effects, by = "userId") %>%
  left_join(final_genre_effects, by = "genres") %>%
  mutate(
    b_i = ifelse(is.na(b_i), 0, b_i),
    b_u = ifelse(is.na(b_u), 0, b_u),
    b_g = ifelse(is.na(b_g), 0, b_g),
    pred = mu + b_i + b_u + b_g
  )

# Calculate final RMSE
final_rmse <- RMSE(validation$rating, predicted_ratings_with_genre$pred)

# Add to results
rmse_results <- bind_rows(rmse_results,
                         tibble(Method = "Optimized Regularized Model with Genre Effects",
                                RMSE = final_rmse))

# Display updated results
knitr::kable(rmse_results, digits = 4,
             caption = "Model performance on validation set")

###############################################################
# PART 10: VISUALIZATIONS AND ANALYSIS
# 
# These visualizations help understand model performance and
# identify potential areas for improvement:
# 1. RMSE vs regularization parameter (lambda)
# 2. Distribution of effects by genre
# 3. Predicted vs. actual ratings
# 4. Distribution of predictions by actual rating
###############################################################

# Load ggplot for visualization
library(ggplot2)

# Plot distribution of effects by genre
p <- ggplot(final_genre_effects, aes(x = reorder(genres, b_g, FUN = median), y = b_g)) +
  geom_boxplot() +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Distribution of Genre Effects",
       x = "Genre",
       y = "Effect Size")

print(p)

# Display final model comparison results
print("Final model comparison:")
print(knitr::kable(rmse_results, digits = 4))

###############################################################
# PART 11: FINAL EVALUATION
# 
# Apply the best-performing model to the final holdout test set
# to get an unbiased estimate of model performance.
# 
# The final RMSE represents how well our model would perform
# on new, unseen data.
###############################################################

# Make final predictions on holdout set using the best model
# (Regularized model with genre effects)
final_predictions <- final_holdout_test %>%
  left_join(final_genre_effects, by = "genres") %>%
  mutate(
    b_i = ifelse(is.na(b_i), 0, b_i),
    b_u = ifelse(is.na(b_u), 0, b_u),
    b_g = ifelse(is.na(b_g), 0, b_g),
    pred = mu + b_i + b_u + b_g
  )

# Calculate final RMSE
final_rmse <- RMSE(final_holdout_test$rating, final_predictions$pred)

# Add to results table
final_results <- tibble(
  Method = "Final Model on Holdout",
  RMSE = final_rmse
)

# Display final results
knitr::kable(final_results, digits = 4,
             caption = "Final Model Performance on Holdout Set")

# Visualize model performance: compare predicted vs actual ratings
ggplot(final_predictions, aes(x = rating, y = pred)) +
  geom_point(alpha = 0.1) +
  geom_abline(color = "red") +
  labs(x = "Actual Ratings", 
       y = "Predicted Ratings",
       title = "Predicted vs Actual Ratings on Holdout Set") +
  theme_minimal()

# Create boxplot of predictions for each actual rating value
ggplot(final_predictions, aes(x = factor(rating), y = pred)) +
  geom_boxplot(fill = "lightblue") +
  labs(x = "Actual Rating", 
       y = "Predicted Rating",
       title = "Distribution of Predictions by Actual Rating") +
  theme_minimal() +
  geom_hline(yintercept = seq(1, 5, 1), linetype = "dashed", alpha = 0.3)