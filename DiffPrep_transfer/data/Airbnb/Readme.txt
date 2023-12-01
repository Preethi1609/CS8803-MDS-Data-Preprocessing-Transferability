Source: This dataset is provided by Jenny Blase.

Description: This dataset contains over 40,000 observations for the top 10 tourist destinations and major US metropolitan areas scraped from Airbnb.com by filtering based on neighborhood. Demographic and economic data were scraped from city-data.com. Airbnb location data didn’t come with zip codes but with coordinates from which we identified the closest zip code and stored it in a new column. With the zip codes for Airbnb listings, we merged the demographics to produce the final table.

Errors: This dataset contains outliers, missing values and duplicates. 

ML tasks: Classification: predict rating = 5 or not