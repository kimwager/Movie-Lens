# Create the datasets as instructed (code provided)

##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

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

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                       stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# All code hereafter was not provided as part of the assessment introduction

#For model development, instead of using the entire `edx` dataset, I will split it into two parts:
#1.  `edx_train` to develop the model
#2.  `validation` to tune and evaluate intermediate model performance


# Create training and validation set from full edx dataset for model development

set.seed(1, sample.kind="Rounding")
validation_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
validation <- edx[validation_index,]
edx_train <- edx[-validation_index,]

# Generate a baseline model

## Step 1: Calculate the mean rating (μ)

# Calculate the overall mean rating from training data
mu <- mean(edx_train$rating)
print(paste("Overall mean rating:", round(mu, 4)))


## Step 2: Create predictions for validation set

# Create predictions using just the mean (mu) from the previous step
predictions <- rep(mu, nrow(validation))


### Step 3: Define RMSE function and calculate error

# Define RMSE function
# Takes two parameters: true ratings (edx dataset) and predicted ratings (final_holdout_test dataset)
# Returns a single number representing prediction accuracy (lower is better)
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Calculate RMSE
# Print the mean rating with 4 decimal places for clarity
# paste() combines text and numeric value into a single string
naive_rmse <- RMSE(validation$rating, predictions)
print(paste("RMSE for baseline model:", round(naive_rmse, 4)))


### Step 4: Create results table

# Create results table
rmse_results <- tibble(
  Method = "Baseline model",
  RMSE = naive_rmse
)

knitr::kable(rmse_results, digits = 4, 
             caption = "Baseline model performance")


## Modelling movie effects

# Calculate global mean (as previoulsy)
mu <- mean(edx_train$rating)

# Calculate movie effect (b_i)
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

# Caluclate improvement over naive RMSE
improvement <- naive_rmse - movie_effect_rmse
improvement_percentage_movie_effects <- (naive_rmse - movie_effect_rmse) / naive_rmse * 100

# Add to results table
rmse_results <- bind_rows(rmse_results,
                         tibble(Method = "Movie Effects Model",
                                RMSE = movie_effect_rmse))


# Modelling user effects
user_avgs <- edx_train %>%
   group_by(userId) %>%
   summarize(b_u = mean(rating - mu))

# Predict ratings
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

# Caluclate improvement over naive RMSE
improvement <- naive_rmse - user_effect_rmse
improvement_percentage_user_effects <- (naive_rmse - user_effect_rmse) / naive_rmse * 100

# Add to results
rmse_results <- bind_rows(rmse_results,
                         tibble(Method = "User Effects Model",
                                RMSE = user_effect_rmse))

# Display results table
knitr::kable(rmse_results, digits = 4,
             caption = "Model performance on validation set")                              


# Combined movie and user effects model
# Calculate global mean (as previously)
mu <- mean(edx_train$rating)

# Calculate movie effects (as previously)
movie_avgs <- edx_train %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# Calculate user effects accounting for movie effects (join movie effects with training data)

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

# Caluclate improvement over naive RMSE
improvement <- naive_rmse - combined_effect_rmse
improvement_percentage_combined_effects <- (naive_rmse - combined_effect_rmse) / naive_rmse * 100

# Add to results
rmse_results <- bind_rows(rmse_results,
                         tibble(Method = "Combined Effects Model",
                                RMSE = combined_effect_rmse))

# Display results table
knitr::kable(rmse_results, digits = 4,
             caption = "Model performance on validation set")

# Create a container list for all model components
movie_model <- list()

# Regularized Movie and User Effects Model
# Try different lambda values
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

# Plot RMSE vs lambda
qplot(movie_model$lambdas, movie_model$rmses) +
  geom_line() +
  xlab("Lambda") +
  ylab("RMSE") +
  ggtitle("RMSE vs Regularization Parameter")

# Add results
rmse_results <- bind_rows(rmse_results,
                         tibble(Method = "Regularized Model",
                                RMSE = min(movie_model$rmses)))

# Display updated results
knitr::kable(rmse_results, digits = 4,
             caption = "Model performance on validation set")

# Calculate regularized genre effects with the same lambda
genre_model <- movie_model # Create a copy to extend with genre effects

genre_model$genre_effects <- edx_train %>%
  left_join(genre_model$movie_effects, by = "movieId") %>%
  left_join(genre_model$user_effects, by = "userId") %>%
  group_by(genres) %>%
  summarize(
    b_g = sum(rating - mu - b_i - b_u)/(n() + genre_model$optimal_lambda),
    n_g = n()
  )

# Make predictions including genre effects
predicted_ratings_with_genre <- validation %>%
  left_join(genre_model$movie_effects, by = "movieId") %>%
  left_join(genre_model$user_effects, by = "userId") %>%
  left_join(genre_model$genre_effects, by = "genres") %>%
  mutate(
    b_i = ifelse(is.na(b_i), 0, b_i),
    b_u = ifelse(is.na(b_u), 0, b_u),
    b_g = ifelse(is.na(b_g), 0, b_g),
    pred = mu + b_i + b_u + b_g
  )

# Calculate RMSE with genre effects
genre_model$rmse <- RMSE(validation$rating, predicted_ratings_with_genre$pred)

# Add results to the comparison table
rmse_results <- bind_rows(rmse_results,
                         tibble(Method = "Regularized Model with Genre Effects",
                                RMSE = genre_model$rmse))

# Display updated results
knitr::kable(rmse_results, digits = 4,
             caption = "Model performance comparison on validation set")

# Optional: Look at the impact of different genres
genre_impact <- genre_model$genre_effects %>%
  arrange(desc(abs(b_g))) %>%
  mutate(b_g = round(b_g, 4)) %>%
  head(10)

print("Top 10 genres by absolute effect size:")
knitr::kable(genre_impact)

# Genre approach 2: Split genres and analyze separately
# Calculate global mean
mu <- mean(edx_train$rating)

# Set lambda directly for simplicity
lambda <- 5  # We can tune this if needed

# Get list of unique genres
all_genres <- unique(unlist(strsplit(edx_train$genres, "\\|")))
print("Number of unique individual genres:")
print(length(all_genres))
print("Unique genres:")
print(all_genres)

# Function to check if a movie has a specific genre
has_genre <- function(genre_string, target_genre) {
  genres <- unlist(strsplit(genre_string, "\\|"))
  return(target_genre %in% genres)
}

# Calculate effect for each genre separately
genre_specific_effects <- data.frame()

for(genre in all_genres) {
  # Calculate effect for this genre
  genre_effect <- edx_train %>%
    mutate(has_genre = sapply(genres, has_genre, target_genre = genre)) %>%
    filter(has_genre) %>%
    group_by(userId) %>%
    summarize(
      effect = sum(rating - mu)/(n() + lambda),
      n = n(),
      genre = genre,
      .groups = 'drop'
    )
  
  genre_specific_effects <- rbind(genre_specific_effects, genre_effect)
}

# Analyze genre effects
genre_summary <- genre_specific_effects %>%
  group_by(genre) %>%
  summarize(
    mean_effect = mean(effect, na.rm = TRUE),
    sd_effect = sd(effect, na.rm = TRUE),
    n_users = n(),
    .groups = 'drop'
  ) %>%
  arrange(desc(abs(mean_effect)))

# Display genre summary
print("Genre effect summary:")
print(knitr::kable(genre_summary, digits = 4))

# Calculate predictions for validation set
validation_predictions <- data.frame()

for(genre in all_genres) {
  # Get effects for this genre
  genre_effects <- genre_specific_effects %>%
    filter(genre == genre) %>%
    select(userId, effect)
  
  # Add predictions for movies with this genre
  genre_pred <- validation %>%
    mutate(has_genre = sapply(genres, has_genre, target_genre = genre)) %>%
    filter(has_genre) %>%
    left_join(genre_effects, by = "userId") %>%
    mutate(
      effect = ifelse(is.na(effect), 0, effect),
      pred = mu + effect
    )
  
  validation_predictions <- rbind(validation_predictions, genre_pred)
}

# Calculate overall RMSE for genre-specific model
genre_rmse <- RMSE(validation_predictions$rating, validation_predictions$pred)

# Add to results
rmse_results <- bind_rows(rmse_results,
                         tibble(Method = "Genre-Specific User Effects",
                                RMSE = genre_rmse))

# Plot distribution of effects by genre
library(ggplot2)
p <- ggplot(genre_specific_effects, aes(x = reorder(genre, effect, FUN = median), y = effect)) +
  geom_boxplot() +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Distribution of User-Genre Effects",
       x = "Genre",
       y = "Effect Size")

print(p)

# Display final results
print("Final model comparison:")
print(knitr::kable(rmse_results, digits = 4))

# Make final predictions on holdout set
final_predictions <- final_holdout_test %>%
  left_join(genre_model$movie_effects, by = "movieId") %>%
  left_join(genre_model$user_effects, by = "userId") %>%
  left_join(genre_model$genre_effects, by = "genres") %>%
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

# Compare predicted vs actual ratings distribution

ggplot(final_predictions, aes(x = rating, y = pred)) +
  geom_point(alpha = 0.1) +
  geom_abline(color = "red") +
  labs(x = "Actual Ratings", 
       y = "Predicted Ratings",
       title = "Predicted vs Actual Ratings on Holdout Set") +
  theme_minimal()

# Boxplot of predictions for each actual rating
ggplot(final_predictions, aes(x = factor(rating), y = pred)) +
  geom_boxplot(fill = "lightblue") +
  labs(x = "Actual Rating", 
       y = "Predicted Rating",
       title = "Distribution of Predictions by Actual Rating") +
  theme_minimal() +
  geom_hline(yintercept = seq(1, 5, 1), linetype = "dashed", alpha = 0.3)

