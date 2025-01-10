

# Find number of rows and columns
str(edx) 

# Find number of zeros in ratings column
sum(edx$rating == 0)

# How many threes?
sum(edx$rating == 3)

# How many movies?
edx %>% 
  distinct(movieId) %>% 
  nrow()

# How many users?

edx %>% 
  distinct(userId) %>% 
  nrow()

# How many movie ratings are in each of the following genres in the edx dataset? Drama: Comedy: Thriller: Romance:

edx %>%
  summarize(
    Drama = sum(str_detect(genres, "Drama")),
    Comedy = sum(str_detect(genres, "Comedy")),
    Thriller = sum(str_detect(genres, "Thriller")),
    Romance = sum(str_detect(genres, "Romance"))
  )

# Which movie has the greatest number of ratings?

  edx %>%
    group_by(title) %>%
    summarise(number_of_ratings = n()) %>%
    arrange(desc(number_of_ratings)) %>%
    slice(1)

# What are the five most given ratings in order from most to least?
edx %>%
  group_by(rating) %>%
  summarise(count = n()) %>%
  arrange(desc(count)) %>%
  head(5)