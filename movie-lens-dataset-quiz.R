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