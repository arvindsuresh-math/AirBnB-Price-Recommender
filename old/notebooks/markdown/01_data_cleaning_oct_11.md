# Airbnb NYC Data Pre-processing & ETL

This notebook implements the end-to-end data pipeline to create the final modeling dataset from raw InsideAirbnb snapshots.

**Objective:** Load all monthly listings snapshots and the full reviews history, clean features, engineer the `estimated_occupancy_rate` sample weight, and produce a single, model-ready `listing-month` panel.

### 0. Setup & Data Loading


```python
import pandas as pd
import numpy as np
import os
import glob

# --- Configuration ---
# Parent directory containing the 'listings-YY-MM.csv' files and '{CITY}-reviews-detailed...csv'
CITY = "nyc"
INPUT_DATA_DIR = os.path.expanduser(f"~/Downloads/insideairbnb/{CITY}") 
OUTPUT_DATA_DIR = os.path.expanduser(f"../data/{CITY}")
OUTPUT_FILENAME = f"{CITY}_dataset_oct_17.parquet"

# Configure pandas display
pd.options.display.max_columns = 100

# --- Load All Monthly Listings Snapshots ---
listings_files = sorted(glob.glob(os.path.join(INPUT_DATA_DIR, 'listings-*.csv')))
if not listings_files:
    raise FileNotFoundError(f"No 'listings-*.csv' files found in {INPUT_DATA_DIR}")

print(f"Found {len(listings_files)} monthly listings files. Loading and concatenating...")

dfs = []
for file in listings_files:
    # low_memory=False handles mixed data types in raw CSVs
    df = pd.read_csv(file, low_memory=False) 
    dfs.append(df)

raw_listings_df = pd.concat(dfs, ignore_index=True)
print(f"Successfully loaded {len(raw_listings_df):,} total listing records.")

# --- Load Full Reviews History ---
reviews_path = os.path.join(INPUT_DATA_DIR, f'{CITY}-reviews-detailed-insideairbnb.csv')
print(f"Loading reviews from: {os.path.basename(reviews_path)}...")
try:
    raw_reviews_df = pd.read_csv(reviews_path)
    print(f"Successfully loaded {len(raw_reviews_df):,} reviews.")
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find reviews file at: {reviews_path}")

# Display samples
print("\nListings Sample:")
display(raw_listings_df.head(2))
print("\nReviews Sample:")
display(raw_reviews_df.head(2))

# Display column info
print("\nListings DataFrame Info:")
print(raw_listings_df.info())
print("\nReviews DataFrame Info:")
print(raw_reviews_df.info())
```

    Found 12 monthly listings files. Loading and concatenating...
    Successfully loaded 443,898 total listing records.
    Loading reviews from: nyc-reviews-detailed-insideairbnb.csv...
    Successfully loaded 986,597 reviews.
    
    Listings Sample:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>listing_url</th>
      <th>scrape_id</th>
      <th>last_scraped</th>
      <th>source</th>
      <th>name</th>
      <th>description</th>
      <th>neighborhood_overview</th>
      <th>picture_url</th>
      <th>host_id</th>
      <th>host_url</th>
      <th>host_name</th>
      <th>host_since</th>
      <th>host_location</th>
      <th>host_about</th>
      <th>host_response_time</th>
      <th>host_response_rate</th>
      <th>host_acceptance_rate</th>
      <th>host_is_superhost</th>
      <th>host_thumbnail_url</th>
      <th>host_picture_url</th>
      <th>host_neighbourhood</th>
      <th>host_listings_count</th>
      <th>host_total_listings_count</th>
      <th>host_verifications</th>
      <th>host_has_profile_pic</th>
      <th>host_identity_verified</th>
      <th>neighbourhood</th>
      <th>neighbourhood_cleansed</th>
      <th>neighbourhood_group_cleansed</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>property_type</th>
      <th>room_type</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bathrooms_text</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>amenities</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>maximum_nights</th>
      <th>minimum_minimum_nights</th>
      <th>maximum_minimum_nights</th>
      <th>minimum_maximum_nights</th>
      <th>maximum_maximum_nights</th>
      <th>minimum_nights_avg_ntm</th>
      <th>maximum_nights_avg_ntm</th>
      <th>calendar_updated</th>
      <th>has_availability</th>
      <th>availability_30</th>
      <th>availability_60</th>
      <th>availability_90</th>
      <th>availability_365</th>
      <th>calendar_last_scraped</th>
      <th>number_of_reviews</th>
      <th>number_of_reviews_ltm</th>
      <th>number_of_reviews_l30d</th>
      <th>first_review</th>
      <th>last_review</th>
      <th>review_scores_rating</th>
      <th>review_scores_accuracy</th>
      <th>review_scores_cleanliness</th>
      <th>review_scores_checkin</th>
      <th>review_scores_communication</th>
      <th>review_scores_location</th>
      <th>review_scores_value</th>
      <th>license</th>
      <th>instant_bookable</th>
      <th>calculated_host_listings_count</th>
      <th>calculated_host_listings_count_entire_homes</th>
      <th>calculated_host_listings_count_private_rooms</th>
      <th>calculated_host_listings_count_shared_rooms</th>
      <th>reviews_per_month</th>
      <th>availability_eoy</th>
      <th>number_of_reviews_ly</th>
      <th>estimated_occupancy_l365d</th>
      <th>estimated_revenue_l365d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2595</td>
      <td>https://www.airbnb.com/rooms/2595</td>
      <td>20241104040953</td>
      <td>2024-11-04</td>
      <td>city scrape</td>
      <td>Skylit Midtown Castle Sanctuary</td>
      <td>Beautiful, spacious skylit studio in the heart...</td>
      <td>Centrally located in the heart of Manhattan ju...</td>
      <td>https://a0.muscache.com/pictures/miso/Hosting-...</td>
      <td>2845</td>
      <td>https://www.airbnb.com/users/show/2845</td>
      <td>Jennifer</td>
      <td>2008-09-09</td>
      <td>Woodstock, NY</td>
      <td>A New Yorker since 2000! My passion is creatin...</td>
      <td>within a day</td>
      <td>90%</td>
      <td>21%</td>
      <td>f</td>
      <td>https://a0.muscache.com/im/pictures/user/50fc5...</td>
      <td>https://a0.muscache.com/im/pictures/user/50fc5...</td>
      <td>Midtown</td>
      <td>7.0</td>
      <td>9.0</td>
      <td>['email', 'phone', 'work_email']</td>
      <td>t</td>
      <td>t</td>
      <td>Neighborhood highlights</td>
      <td>Midtown</td>
      <td>Manhattan</td>
      <td>40.75356</td>
      <td>-73.98559</td>
      <td>Entire rental unit</td>
      <td>Entire home/apt</td>
      <td>1</td>
      <td>1.0</td>
      <td>1 bath</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>["Fire extinguisher", "Smoke alarm", "Stove", ...</td>
      <td>$240.00</td>
      <td>30</td>
      <td>1125</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>1125.0</td>
      <td>1125.0</td>
      <td>30.0</td>
      <td>1125.0</td>
      <td>NaN</td>
      <td>t</td>
      <td>30</td>
      <td>60</td>
      <td>90</td>
      <td>365</td>
      <td>2024-11-04</td>
      <td>49</td>
      <td>0</td>
      <td>0</td>
      <td>2009-11-21</td>
      <td>2022-06-21</td>
      <td>4.68</td>
      <td>4.73</td>
      <td>4.63</td>
      <td>4.77</td>
      <td>4.8</td>
      <td>4.81</td>
      <td>4.40</td>
      <td>NaN</td>
      <td>f</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.27</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6848</td>
      <td>https://www.airbnb.com/rooms/6848</td>
      <td>20241104040953</td>
      <td>2024-11-04</td>
      <td>city scrape</td>
      <td>Only 2 stops to Manhattan studio</td>
      <td>Comfortable studio apartment with super comfor...</td>
      <td>NaN</td>
      <td>https://a0.muscache.com/pictures/e4f031a7-f146...</td>
      <td>15991</td>
      <td>https://www.airbnb.com/users/show/15991</td>
      <td>Allen &amp; Irina</td>
      <td>2009-05-06</td>
      <td>New York, NY</td>
      <td>We love to travel. When we travel we like to s...</td>
      <td>within a few hours</td>
      <td>100%</td>
      <td>100%</td>
      <td>t</td>
      <td>https://a0.muscache.com/im/users/15991/profile...</td>
      <td>https://a0.muscache.com/im/users/15991/profile...</td>
      <td>Williamsburg</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>['email', 'phone']</td>
      <td>t</td>
      <td>t</td>
      <td>NaN</td>
      <td>Williamsburg</td>
      <td>Brooklyn</td>
      <td>40.70935</td>
      <td>-73.95342</td>
      <td>Entire rental unit</td>
      <td>Entire home/apt</td>
      <td>3</td>
      <td>1.0</td>
      <td>1 bath</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>["Fire extinguisher", "Smoke alarm", "Stove", ...</td>
      <td>$83.00</td>
      <td>30</td>
      <td>120</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>120.0</td>
      <td>120.0</td>
      <td>30.0</td>
      <td>120.0</td>
      <td>NaN</td>
      <td>t</td>
      <td>0</td>
      <td>15</td>
      <td>15</td>
      <td>185</td>
      <td>2024-11-04</td>
      <td>195</td>
      <td>4</td>
      <td>1</td>
      <td>2009-05-25</td>
      <td>2024-10-05</td>
      <td>4.58</td>
      <td>4.59</td>
      <td>4.85</td>
      <td>4.85</td>
      <td>4.8</td>
      <td>4.69</td>
      <td>4.58</td>
      <td>NaN</td>
      <td>f</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1.04</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    
    Reviews Sample:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>listing_id</th>
      <th>id</th>
      <th>date</th>
      <th>reviewer_id</th>
      <th>reviewer_name</th>
      <th>comments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2595</td>
      <td>17857</td>
      <td>2009-11-21</td>
      <td>50679</td>
      <td>Jean</td>
      <td>Notre séjour de trois nuits.\r&lt;br/&gt;Nous avons ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2595</td>
      <td>19176</td>
      <td>2009-12-05</td>
      <td>53267</td>
      <td>Cate</td>
      <td>Great experience.</td>
    </tr>
  </tbody>
</table>
</div>


    
    Listings DataFrame Info:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 443898 entries, 0 to 443897
    Data columns (total 79 columns):
     #   Column                                        Non-Null Count   Dtype  
    ---  ------                                        --------------   -----  
     0   id                                            443898 non-null  int64  
     1   listing_url                                   443898 non-null  object 
     2   scrape_id                                     443898 non-null  int64  
     3   last_scraped                                  443898 non-null  object 
     4   source                                        443898 non-null  object 
     5   name                                          443874 non-null  object 
     6   description                                   432027 non-null  object 
     7   neighborhood_overview                         236590 non-null  object 
     8   picture_url                                   443891 non-null  object 
     9   host_id                                       443898 non-null  int64  
     10  host_url                                      443898 non-null  object 
     11  host_name                                     441463 non-null  object 
     12  host_since                                    441460 non-null  object 
     13  host_location                                 348943 non-null  object 
     14  host_about                                    252528 non-null  object 
     15  host_response_time                            259303 non-null  object 
     16  host_response_rate                            259303 non-null  object 
     17  host_acceptance_rate                          265377 non-null  object 
     18  host_is_superhost                             438729 non-null  object 
     19  host_thumbnail_url                            441460 non-null  object 
     20  host_picture_url                              441460 non-null  object 
     21  host_neighbourhood                            352281 non-null  object 
     22  host_listings_count                           441460 non-null  float64
     23  host_total_listings_count                     441460 non-null  float64
     24  host_verifications                            441460 non-null  object 
     25  host_has_profile_pic                          441460 non-null  object 
     26  host_identity_verified                        441460 non-null  object 
     27  neighbourhood                                 236602 non-null  object 
     28  neighbourhood_cleansed                        443898 non-null  object 
     29  neighbourhood_group_cleansed                  443898 non-null  object 
     30  latitude                                      443898 non-null  float64
     31  longitude                                     443898 non-null  float64
     32  property_type                                 443898 non-null  object 
     33  room_type                                     443898 non-null  object 
     34  accommodates                                  443898 non-null  int64  
     35  bathrooms                                     267049 non-null  float64
     36  bathrooms_text                                443238 non-null  object 
     37  bedrooms                                      372494 non-null  float64
     38  beds                                          265841 non-null  float64
     39  amenities                                     443898 non-null  object 
     40  price                                         265091 non-null  object 
     41  minimum_nights                                443898 non-null  int64  
     42  maximum_nights                                443898 non-null  int64  
     43  minimum_minimum_nights                        443819 non-null  float64
     44  maximum_minimum_nights                        443819 non-null  float64
     45  minimum_maximum_nights                        443819 non-null  float64
     46  maximum_maximum_nights                        443819 non-null  float64
     47  minimum_nights_avg_ntm                        443839 non-null  float64
     48  maximum_nights_avg_ntm                        443839 non-null  float64
     49  calendar_updated                              0 non-null       float64
     50  has_availability                              377507 non-null  object 
     51  availability_30                               443898 non-null  int64  
     52  availability_60                               443898 non-null  int64  
     53  availability_90                               443898 non-null  int64  
     54  availability_365                              443898 non-null  int64  
     55  calendar_last_scraped                         443898 non-null  object 
     56  number_of_reviews                             443898 non-null  int64  
     57  number_of_reviews_ltm                         443898 non-null  int64  
     58  number_of_reviews_l30d                        443898 non-null  int64  
     59  first_review                                  305391 non-null  object 
     60  last_review                                   305391 non-null  object 
     61  review_scores_rating                          305391 non-null  float64
     62  review_scores_accuracy                        305244 non-null  float64
     63  review_scores_cleanliness                     305357 non-null  float64
     64  review_scores_checkin                         305196 non-null  float64
     65  review_scores_communication                   305297 non-null  float64
     66  review_scores_location                        305161 non-null  float64
     67  review_scores_value                           305172 non-null  float64
     68  license                                       64550 non-null   object 
     69  instant_bookable                              443898 non-null  object 
     70  calculated_host_listings_count                443898 non-null  int64  
     71  calculated_host_listings_count_entire_homes   443898 non-null  int64  
     72  calculated_host_listings_count_private_rooms  443898 non-null  int64  
     73  calculated_host_listings_count_shared_rooms   443898 non-null  int64  
     74  reviews_per_month                             305391 non-null  float64
     75  availability_eoy                              330758 non-null  float64
     76  number_of_reviews_ly                          330758 non-null  float64
     77  estimated_occupancy_l365d                     330758 non-null  float64
     78  estimated_revenue_l365d                       195996 non-null  float64
    dtypes: float64(26), int64(17), object(36)
    memory usage: 267.5+ MB
    None
    
    Reviews DataFrame Info:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 986597 entries, 0 to 986596
    Data columns (total 6 columns):
     #   Column         Non-Null Count   Dtype 
    ---  ------         --------------   ----- 
     0   listing_id     986597 non-null  int64 
     1   id             986597 non-null  int64 
     2   date           986597 non-null  object
     3   reviewer_id    986597 non-null  int64 
     4   reviewer_name  986594 non-null  object
     5   comments       986338 non-null  object
    dtypes: int64(3), object(3)
    memory usage: 45.2+ MB
    None


### 1. Remove unnecessary columns


```python
cols_to_keep = [
    'id',
    'host_id',
    'name',
    'description',
    'host_is_superhost',
    'neighbourhood_cleansed',
    'latitude',
    'longitude',
    'property_type',
    'room_type',
    'accommodates',
    'bathrooms',
    'bedrooms',
    'beds',
    'amenities',
    'minimum_nights',
    'review_scores_rating',  #float
    'review_scores_accuracy',  #float
    'review_scores_cleanliness',  #float
    'review_scores_checkin',  #float
    'review_scores_communication',  #float
    'review_scores_location',  #float
    'review_scores_value',  #float
    'last_scraped',
    'price'
    ]

listings_df = raw_listings_df[cols_to_keep].copy()
print(f"\nReduced listings DataFrame to {len(listings_df.columns)} columns.")
print(listings_df.info())
```

    
    Reduced listings DataFrame to 25 columns.
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 443898 entries, 0 to 443897
    Data columns (total 25 columns):
     #   Column                       Non-Null Count   Dtype  
    ---  ------                       --------------   -----  
     0   id                           443898 non-null  int64  
     1   host_id                      443898 non-null  int64  
     2   name                         443874 non-null  object 
     3   description                  432027 non-null  object 
     4   host_is_superhost            438729 non-null  object 
     5   neighbourhood_cleansed       443898 non-null  object 
     6   latitude                     443898 non-null  float64
     7   longitude                    443898 non-null  float64
     8   property_type                443898 non-null  object 
     9   room_type                    443898 non-null  object 
     10  accommodates                 443898 non-null  int64  
     11  bathrooms                    267049 non-null  float64
     12  bedrooms                     372494 non-null  float64
     13  beds                         265841 non-null  float64
     14  amenities                    443898 non-null  object 
     15  minimum_nights               443898 non-null  int64  
     16  review_scores_rating         305391 non-null  float64
     17  review_scores_accuracy       305244 non-null  float64
     18  review_scores_cleanliness    305357 non-null  float64
     19  review_scores_checkin        305196 non-null  float64
     20  review_scores_communication  305297 non-null  float64
     21  review_scores_location       305161 non-null  float64
     22  review_scores_value          305172 non-null  float64
     23  last_scraped                 443898 non-null  object 
     24  price                        265091 non-null  object 
    dtypes: float64(12), int64(4), object(9)
    memory usage: 84.7+ MB
    None


### 2. Convert the scrape-date to month (1-12), convert `host_is_superhost` col to numeric 0/1


```python
# Convert last_scraped to datetime
raw_listings_df['last_scraped'] = pd.to_datetime(raw_listings_df['last_scraped'], errors='coerce')

# Convert last_scraped to month only (no year)
listings_df['month'] = raw_listings_df['last_scraped'].dt.month

# Drop the last_scraped column as it's no longer needed
listings_df = listings_df.drop(columns=['last_scraped'])
```

### 3. Clean price column, drop outliers, add price-per-person and log1p of both


```python
# Convert prices to float
listings_df['price'] = listings_df['price'].replace(r'[\$,]', '', regex=True).astype(float)

# Drop NaN's from price column and make it float
listings_df = listings_df.dropna(subset=['price'])

# Add price_per_person column
listings_df['price_per_person'] = listings_df['price'] / listings_df['accommodates']

# Drop the bottom 1% and top 2% of price_per_person to remove outliers
lower_bound = listings_df['price_per_person'].quantile(0.01)
upper_bound = listings_df['price_per_person'].quantile(0.98)
listings_df = listings_df[(listings_df['price_per_person'] >= lower_bound) & (listings_df['price_per_person'] <= upper_bound)]

# Add log1p transformed columns
listings_df['log_price'] = np.log1p(listings_df['price'])
listings_df['log_price_per_person'] = np.log1p(listings_df['price_per_person'])

# Print info and a sample
print("\nUpdated Listings DataFrame Info:")
print(listings_df.info())
print("\nListings DataFrame Sample with New Columns:")
display(listings_df.head(5))
```

    
    Updated Listings DataFrame Info:
    <class 'pandas.core.frame.DataFrame'>
    Index: 257185 entries, 0 to 443897
    Data columns (total 28 columns):
     #   Column                       Non-Null Count   Dtype  
    ---  ------                       --------------   -----  
     0   id                           257185 non-null  int64  
     1   host_id                      257185 non-null  int64  
     2   name                         257185 non-null  object 
     3   description                  252584 non-null  object 
     4   host_is_superhost            253072 non-null  object 
     5   neighbourhood_cleansed       257185 non-null  object 
     6   latitude                     257185 non-null  float64
     7   longitude                    257185 non-null  float64
     8   property_type                257185 non-null  object 
     9   room_type                    257185 non-null  object 
     10  accommodates                 257185 non-null  int64  
     11  bathrooms                    257107 non-null  float64
     12  bedrooms                     256403 non-null  float64
     13  beds                         256269 non-null  float64
     14  amenities                    257185 non-null  object 
     15  minimum_nights               257185 non-null  int64  
     16  review_scores_rating         181445 non-null  float64
     17  review_scores_accuracy       181427 non-null  float64
     18  review_scores_cleanliness    181427 non-null  float64
     19  review_scores_checkin        181427 non-null  float64
     20  review_scores_communication  181427 non-null  float64
     21  review_scores_location       181418 non-null  float64
     22  review_scores_value          181427 non-null  float64
     23  price                        257185 non-null  float64
     24  month                        257185 non-null  int32  
     25  price_per_person             257185 non-null  float64
     26  log_price                    257185 non-null  float64
     27  log_price_per_person         257185 non-null  float64
    dtypes: float64(16), int32(1), int64(4), object(7)
    memory usage: 55.9+ MB
    None
    
    Listings DataFrame Sample with New Columns:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>host_id</th>
      <th>name</th>
      <th>description</th>
      <th>host_is_superhost</th>
      <th>neighbourhood_cleansed</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>property_type</th>
      <th>room_type</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>amenities</th>
      <th>minimum_nights</th>
      <th>review_scores_rating</th>
      <th>review_scores_accuracy</th>
      <th>review_scores_cleanliness</th>
      <th>review_scores_checkin</th>
      <th>review_scores_communication</th>
      <th>review_scores_location</th>
      <th>review_scores_value</th>
      <th>price</th>
      <th>month</th>
      <th>price_per_person</th>
      <th>log_price</th>
      <th>log_price_per_person</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2595</td>
      <td>2845</td>
      <td>Skylit Midtown Castle Sanctuary</td>
      <td>Beautiful, spacious skylit studio in the heart...</td>
      <td>f</td>
      <td>Midtown</td>
      <td>40.75356</td>
      <td>-73.98559</td>
      <td>Entire rental unit</td>
      <td>Entire home/apt</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>["Fire extinguisher", "Smoke alarm", "Stove", ...</td>
      <td>30</td>
      <td>4.68</td>
      <td>4.73</td>
      <td>4.63</td>
      <td>4.77</td>
      <td>4.80</td>
      <td>4.81</td>
      <td>4.40</td>
      <td>240.0</td>
      <td>11</td>
      <td>240.000000</td>
      <td>5.484797</td>
      <td>5.484797</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6848</td>
      <td>15991</td>
      <td>Only 2 stops to Manhattan studio</td>
      <td>Comfortable studio apartment with super comfor...</td>
      <td>t</td>
      <td>Williamsburg</td>
      <td>40.70935</td>
      <td>-73.95342</td>
      <td>Entire rental unit</td>
      <td>Entire home/apt</td>
      <td>3</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>["Fire extinguisher", "Smoke alarm", "Stove", ...</td>
      <td>30</td>
      <td>4.58</td>
      <td>4.59</td>
      <td>4.85</td>
      <td>4.85</td>
      <td>4.80</td>
      <td>4.69</td>
      <td>4.58</td>
      <td>83.0</td>
      <td>11</td>
      <td>27.666667</td>
      <td>4.430817</td>
      <td>3.355735</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6872</td>
      <td>16104</td>
      <td>Uptown Sanctuary w/ Private Bath (Month to Month)</td>
      <td>This charming distancing-friendly month-to-mon...</td>
      <td>f</td>
      <td>East Harlem</td>
      <td>40.80107</td>
      <td>-73.94255</td>
      <td>Private room in condo</td>
      <td>Private room</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>["Heating", "Washer", "Fire extinguisher", "Sm...</td>
      <td>30</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>65.0</td>
      <td>11</td>
      <td>65.000000</td>
      <td>4.189655</td>
      <td>4.189655</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6990</td>
      <td>16800</td>
      <td>UES Beautiful Blue Room</td>
      <td>Beautiful peaceful healthy home</td>
      <td>t</td>
      <td>East Harlem</td>
      <td>40.78778</td>
      <td>-73.94759</td>
      <td>Private room in rental unit</td>
      <td>Private room</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>["Fire extinguisher", "Smoke alarm", "Stove", ...</td>
      <td>30</td>
      <td>4.88</td>
      <td>4.83</td>
      <td>4.95</td>
      <td>4.96</td>
      <td>4.95</td>
      <td>4.85</td>
      <td>4.85</td>
      <td>71.0</td>
      <td>11</td>
      <td>71.000000</td>
      <td>4.276666</td>
      <td>4.276666</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7097</td>
      <td>17571</td>
      <td>Perfect for Your Parents, With Garden &amp; Patio</td>
      <td>Parents/grandparents coming to town or are you...</td>
      <td>t</td>
      <td>Fort Greene</td>
      <td>40.69194</td>
      <td>-73.97389</td>
      <td>Private room in guest suite</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>["Fire extinguisher", "Smoke alarm", "Private ...</td>
      <td>2</td>
      <td>4.89</td>
      <td>4.91</td>
      <td>4.89</td>
      <td>4.96</td>
      <td>4.93</td>
      <td>4.95</td>
      <td>4.82</td>
      <td>205.0</td>
      <td>11</td>
      <td>102.500000</td>
      <td>5.327876</td>
      <td>4.639572</td>
    </tr>
  </tbody>
</table>
</div>


### 4. Keep only listings with at least one review, drop rows with NaN's, keep only listings that appear in at least 5 months


```python
# Compare IDs between listings_df and raw_reviews_df
listings_ids = set(listings_df['id'].unique())
reviews_ids = set(raw_reviews_df['listing_id'].unique())

common_ids = listings_ids & reviews_ids
only_in_listings = listings_ids - reviews_ids
only_in_reviews = reviews_ids - listings_ids

print(f"Total unique IDs in listings: {len(listings_ids)}")
print(f"Total unique IDs in reviews: {len(reviews_ids)}")
print(f"Common IDs: {len(common_ids)}")
print(f"IDs only in listings: {len(only_in_listings)}")
print(f"IDs only in reviews: {len(only_in_reviews)}")

# Optionally, display some samples
print("\nSample common IDs:", list(common_ids)[:5])
print("Sample only in listings:", list(only_in_listings)[:5])
print("Sample only in reviews:", list(only_in_reviews)[:5])

# Keep only common IDs in listings and reviews
common_listings_df = listings_df[listings_df['id'].isin(common_ids)]
common_reviews_df = raw_reviews_df[raw_reviews_df['listing_id'].isin(common_ids)]

# Drop all listings with NaN's
common_listings_df = common_listings_df.dropna()

# Keep only listings that appear at least 5 times
common_listings_df = common_listings_df[common_listings_df.groupby('id')['id'].transform('size') >= 5]

# Display info after filtering
print("\nFiltered Listings DataFrame Info:")
print(common_listings_df.info())
print("\nFiltered Reviews DataFrame Info:")
print(common_reviews_df.info())
```

    Total unique IDs in listings: 32995
    Total unique IDs in reviews: 24923
    Common IDs: 16966
    IDs only in listings: 16029
    IDs only in reviews: 7957
    
    Sample common IDs: [np.int64(48758785), np.int64(1194458269999890435), np.int64(847115675515322373), np.int64(819206), np.int64(2949128)]
    Sample only in listings: [np.int64(681805560172871680), np.int64(649151763894140930), np.int64(1472518013221634050), np.int64(1505181786944995332), np.int64(43679750)]
    Sample only in reviews: [np.int64(1277955), np.int64(952770872245256198), np.int64(491529), np.int64(14155795), np.int64(2261018)]
    
    Filtered Listings DataFrame Info:
    <class 'pandas.core.frame.DataFrame'>
    Index: 148998 entries, 0 to 442318
    Data columns (total 28 columns):
     #   Column                       Non-Null Count   Dtype  
    ---  ------                       --------------   -----  
     0   id                           148998 non-null  int64  
     1   host_id                      148998 non-null  int64  
     2   name                         148998 non-null  object 
     3   description                  148998 non-null  object 
     4   host_is_superhost            148998 non-null  object 
     5   neighbourhood_cleansed       148998 non-null  object 
     6   latitude                     148998 non-null  float64
     7   longitude                    148998 non-null  float64
     8   property_type                148998 non-null  object 
     9   room_type                    148998 non-null  object 
     10  accommodates                 148998 non-null  int64  
     11  bathrooms                    148998 non-null  float64
     12  bedrooms                     148998 non-null  float64
     13  beds                         148998 non-null  float64
     14  amenities                    148998 non-null  object 
     15  minimum_nights               148998 non-null  int64  
     16  review_scores_rating         148998 non-null  float64
     17  review_scores_accuracy       148998 non-null  float64
     18  review_scores_cleanliness    148998 non-null  float64
     19  review_scores_checkin        148998 non-null  float64
     20  review_scores_communication  148998 non-null  float64
     21  review_scores_location       148998 non-null  float64
     22  review_scores_value          148998 non-null  float64
     23  price                        148998 non-null  float64
     24  month                        148998 non-null  int32  
     25  price_per_person             148998 non-null  float64
     26  log_price                    148998 non-null  float64
     27  log_price_per_person         148998 non-null  float64
    dtypes: float64(16), int32(1), int64(4), object(7)
    memory usage: 32.4+ MB
    None
    
    Filtered Reviews DataFrame Info:
    <class 'pandas.core.frame.DataFrame'>
    Index: 852174 entries, 0 to 986596
    Data columns (total 6 columns):
     #   Column         Non-Null Count   Dtype 
    ---  ------         --------------   ----- 
     0   listing_id     852174 non-null  int64 
     1   id             852174 non-null  int64 
     2   date           852174 non-null  object
     3   reviewer_id    852174 non-null  int64 
     4   reviewer_name  852173 non-null  object
     5   comments       851940 non-null  object
    dtypes: int64(3), object(3)
    memory usage: 45.5+ MB
    None


### 5. Add column with total reviews extracted from `common_reviews_df`, format `host_is_superhost`, `bedrooms`, and `beds` columns 


```python
# Aggregate reviews to get total reviews per listing
reviews_count = common_reviews_df.groupby('listing_id').size().reset_index(name='total_reviews')

# Merge on the listing ID
final_df = common_listings_df.merge(reviews_count, left_on='id', right_on='listing_id', how='left')

# Convert total_reviews to int
final_df['total_reviews'] = final_df['total_reviews'].astype('int')

# Drop the redundant listing_id column
final_df = final_df.drop(columns=['listing_id'])

# Convert host_is_superhost to numeric 0/1
final_df['host_is_superhost'] = final_df['host_is_superhost'].astype(str).map({'t': 1, 'f': 0})

# Convert bedrooms and beds to int
final_df['bedrooms'] = final_df['bedrooms'].astype('int')
final_df['beds'] = final_df['beds'].astype('int')

# Print information about the final DataFrame
print(f"\nFinal listings dataset for {CITY}:")
display(final_df.info())

# Display 3 sample listings (all occurrences)
sample_ids = np.random.choice(final_df['id'].unique(), size=3, replace=False)
for listing_id in sample_ids:
    listing_reviews = final_df[final_df['id'] == listing_id]
    print(f"\nSample data for listing ID {listing_id}:")
    display(listing_reviews)
```

    
    Final listings dataset for nyc:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 148998 entries, 0 to 148997
    Data columns (total 29 columns):
     #   Column                       Non-Null Count   Dtype  
    ---  ------                       --------------   -----  
     0   id                           148998 non-null  int64  
     1   host_id                      148998 non-null  int64  
     2   name                         148998 non-null  object 
     3   description                  148998 non-null  object 
     4   host_is_superhost            148998 non-null  int64  
     5   neighbourhood_cleansed       148998 non-null  object 
     6   latitude                     148998 non-null  float64
     7   longitude                    148998 non-null  float64
     8   property_type                148998 non-null  object 
     9   room_type                    148998 non-null  object 
     10  accommodates                 148998 non-null  int64  
     11  bathrooms                    148998 non-null  float64
     12  bedrooms                     148998 non-null  int64  
     13  beds                         148998 non-null  int64  
     14  amenities                    148998 non-null  object 
     15  minimum_nights               148998 non-null  int64  
     16  review_scores_rating         148998 non-null  float64
     17  review_scores_accuracy       148998 non-null  float64
     18  review_scores_cleanliness    148998 non-null  float64
     19  review_scores_checkin        148998 non-null  float64
     20  review_scores_communication  148998 non-null  float64
     21  review_scores_location       148998 non-null  float64
     22  review_scores_value          148998 non-null  float64
     23  price                        148998 non-null  float64
     24  month                        148998 non-null  int32  
     25  price_per_person             148998 non-null  float64
     26  log_price                    148998 non-null  float64
     27  log_price_per_person         148998 non-null  float64
     28  total_reviews                148998 non-null  int64  
    dtypes: float64(14), int32(1), int64(8), object(6)
    memory usage: 32.4+ MB



    None


    
    Sample data for listing ID 7787799:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>host_id</th>
      <th>name</th>
      <th>description</th>
      <th>host_is_superhost</th>
      <th>neighbourhood_cleansed</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>property_type</th>
      <th>room_type</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>amenities</th>
      <th>minimum_nights</th>
      <th>review_scores_rating</th>
      <th>review_scores_accuracy</th>
      <th>review_scores_cleanliness</th>
      <th>review_scores_checkin</th>
      <th>review_scores_communication</th>
      <th>review_scores_location</th>
      <th>review_scores_value</th>
      <th>price</th>
      <th>month</th>
      <th>price_per_person</th>
      <th>log_price</th>
      <th>log_price_per_person</th>
      <th>total_reviews</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1105</th>
      <td>7787799</td>
      <td>26377263</td>
      <td>Home For Medical Professionals - "Belladonna"</td>
      <td>STAT Living LLC is the #1 Trusted Company prov...</td>
      <td>1</td>
      <td>Bushwick</td>
      <td>40.70513</td>
      <td>-73.91937</td>
      <td>Private room in rental unit</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Stainless steel stove", "Lock on bedroom doo...</td>
      <td>30</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>65.0</td>
      <td>11</td>
      <td>32.5</td>
      <td>4.189655</td>
      <td>3.511545</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12933</th>
      <td>7787799</td>
      <td>26377263</td>
      <td>Home For Medical Professionals - "Belladonna"</td>
      <td>STAT Living LLC is the #1 Trusted Company prov...</td>
      <td>1</td>
      <td>Bushwick</td>
      <td>40.70513</td>
      <td>-73.91937</td>
      <td>Private room in rental unit</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Paid dryer \u2013 In building", "Freezer", "...</td>
      <td>30</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>65.0</td>
      <td>12</td>
      <td>32.5</td>
      <td>4.189655</td>
      <td>3.511545</td>
      <td>2</td>
    </tr>
    <tr>
      <th>24873</th>
      <td>7787799</td>
      <td>26377263</td>
      <td>Home For Medical Professionals - "Belladonna"</td>
      <td>STAT Living LLC is the #1 Trusted Company prov...</td>
      <td>1</td>
      <td>Bushwick</td>
      <td>40.70513</td>
      <td>-73.91937</td>
      <td>Private room in rental unit</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Hangers", "Coffee maker", "Smoke alarm", "HD...</td>
      <td>30</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>65.0</td>
      <td>1</td>
      <td>32.5</td>
      <td>4.189655</td>
      <td>3.511545</td>
      <td>2</td>
    </tr>
    <tr>
      <th>36926</th>
      <td>7787799</td>
      <td>26377263</td>
      <td>Home For Medical Professionals - "Belladonna"</td>
      <td>STAT Living LLC is the #1 Trusted Company prov...</td>
      <td>1</td>
      <td>Bushwick</td>
      <td>40.70513</td>
      <td>-73.91937</td>
      <td>Private room in rental unit</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Hot water", "Bathtub", "Free street parking"...</td>
      <td>30</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>65.0</td>
      <td>2</td>
      <td>32.5</td>
      <td>4.189655</td>
      <td>3.511545</td>
      <td>2</td>
    </tr>
    <tr>
      <th>49220</th>
      <td>7787799</td>
      <td>26377263</td>
      <td>Home For Medical Professionals - "Belladonna"</td>
      <td>STAT Living LLC is the #1 Trusted Company prov...</td>
      <td>1</td>
      <td>Bushwick</td>
      <td>40.70513</td>
      <td>-73.91937</td>
      <td>Private room in rental unit</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Laundromat nearby", "Paid dryer \u2013 In bu...</td>
      <td>30</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>59.0</td>
      <td>3</td>
      <td>29.5</td>
      <td>4.094345</td>
      <td>3.417727</td>
      <td>2</td>
    </tr>
    <tr>
      <th>61408</th>
      <td>7787799</td>
      <td>26377263</td>
      <td>Home For Medical Professionals - "Belladonna"</td>
      <td>STAT Living LLC is the #1 Trusted Company prov...</td>
      <td>1</td>
      <td>Bushwick</td>
      <td>40.70513</td>
      <td>-73.91937</td>
      <td>Private room in rental unit</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Microwave", "Free street parking", "Stainles...</td>
      <td>30</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>59.0</td>
      <td>4</td>
      <td>29.5</td>
      <td>4.094345</td>
      <td>3.417727</td>
      <td>2</td>
    </tr>
    <tr>
      <th>73851</th>
      <td>7787799</td>
      <td>26377263</td>
      <td>Home For Medical Professionals - "Belladonna"</td>
      <td>STAT Living LLC is the #1 Trusted Company prov...</td>
      <td>1</td>
      <td>Bushwick</td>
      <td>40.70513</td>
      <td>-73.91937</td>
      <td>Private room in rental unit</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Iron", "Dining table", "Lock on bedroom door...</td>
      <td>30</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>59.0</td>
      <td>5</td>
      <td>29.5</td>
      <td>4.094345</td>
      <td>3.417727</td>
      <td>2</td>
    </tr>
    <tr>
      <th>86296</th>
      <td>7787799</td>
      <td>26377263</td>
      <td>Home For Medical Professionals - "Belladonna"</td>
      <td>STAT Living LLC is the #1 Trusted Company prov...</td>
      <td>1</td>
      <td>Bushwick</td>
      <td>40.70513</td>
      <td>-73.91937</td>
      <td>Private room in rental unit</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Dedicated workspace", "Essentials", "Stainle...</td>
      <td>30</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>59.0</td>
      <td>6</td>
      <td>29.5</td>
      <td>4.094345</td>
      <td>3.417727</td>
      <td>2</td>
    </tr>
    <tr>
      <th>99066</th>
      <td>7787799</td>
      <td>26377263</td>
      <td>Home For Medical Professionals - "Belladonna"</td>
      <td>STAT Living LLC is the #1 Trusted Company prov...</td>
      <td>1</td>
      <td>Bushwick</td>
      <td>40.70513</td>
      <td>-73.91937</td>
      <td>Private room in rental unit</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Paid dryer \u2013 In building", "Bathtub", "...</td>
      <td>30</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>67.0</td>
      <td>7</td>
      <td>33.5</td>
      <td>4.219508</td>
      <td>3.540959</td>
      <td>2</td>
    </tr>
    <tr>
      <th>111829</th>
      <td>7787799</td>
      <td>26377263</td>
      <td>Home For Medical Professionals - "Belladonna"</td>
      <td>STAT Living LLC is the #1 Trusted Company prov...</td>
      <td>1</td>
      <td>Bushwick</td>
      <td>40.70513</td>
      <td>-73.91937</td>
      <td>Private room in rental unit</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Hot water kettle", "Microwave", "Paid washer...</td>
      <td>30</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>59.0</td>
      <td>8</td>
      <td>29.5</td>
      <td>4.094345</td>
      <td>3.417727</td>
      <td>2</td>
    </tr>
    <tr>
      <th>124515</th>
      <td>7787799</td>
      <td>26377263</td>
      <td>Home For Medical Professionals - "Belladonna"</td>
      <td>STAT Living LLC is the #1 Trusted Company prov...</td>
      <td>1</td>
      <td>Bushwick</td>
      <td>40.70513</td>
      <td>-73.91937</td>
      <td>Private room in rental unit</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Toaster", "Wifi", "Stainless steel stove", "...</td>
      <td>30</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>64.0</td>
      <td>9</td>
      <td>32.0</td>
      <td>4.174387</td>
      <td>3.496508</td>
      <td>2</td>
    </tr>
    <tr>
      <th>137271</th>
      <td>7787799</td>
      <td>26377263</td>
      <td>Home For Medical Professionals - "Belladonna"</td>
      <td>STAT Living LLC is the #1 Trusted Company prov...</td>
      <td>1</td>
      <td>Bushwick</td>
      <td>40.70513</td>
      <td>-73.91937</td>
      <td>Private room in rental unit</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Paid washer \u2013 In building", "Dedicated ...</td>
      <td>30</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.5</td>
      <td>56.0</td>
      <td>10</td>
      <td>28.0</td>
      <td>4.043051</td>
      <td>3.367296</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>


    
    Sample data for listing ID 742712860134731702:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>host_id</th>
      <th>name</th>
      <th>description</th>
      <th>host_is_superhost</th>
      <th>neighbourhood_cleansed</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>property_type</th>
      <th>room_type</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>amenities</th>
      <th>minimum_nights</th>
      <th>review_scores_rating</th>
      <th>review_scores_accuracy</th>
      <th>review_scores_cleanliness</th>
      <th>review_scores_checkin</th>
      <th>review_scores_communication</th>
      <th>review_scores_location</th>
      <th>review_scores_value</th>
      <th>price</th>
      <th>month</th>
      <th>price_per_person</th>
      <th>log_price</th>
      <th>log_price_per_person</th>
      <th>total_reviews</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8010</th>
      <td>742712860134731702</td>
      <td>113295877</td>
      <td>3rd fl. loft style bedroom in Family friendly ...</td>
      <td>Take it easy at this unique and tranquil getaw...</td>
      <td>1</td>
      <td>Tompkinsville</td>
      <td>40.63636</td>
      <td>-74.08908</td>
      <td>Private room in rental unit</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Lock on bedroom door", "Smoke alarm", "Gas s...</td>
      <td>30</td>
      <td>4.67</td>
      <td>4.83</td>
      <td>4.83</td>
      <td>4.83</td>
      <td>4.67</td>
      <td>4.33</td>
      <td>4.67</td>
      <td>40.0</td>
      <td>11</td>
      <td>20.0</td>
      <td>3.713572</td>
      <td>3.044522</td>
      <td>6</td>
    </tr>
    <tr>
      <th>19974</th>
      <td>742712860134731702</td>
      <td>113295877</td>
      <td>3rd fl. loft style bedroom in Family friendly ...</td>
      <td>Take it easy at this unique and tranquil getaw...</td>
      <td>1</td>
      <td>Tompkinsville</td>
      <td>40.63636</td>
      <td>-74.08908</td>
      <td>Private room in rental unit</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Freezer", "Free street parking", "Private ba...</td>
      <td>30</td>
      <td>4.67</td>
      <td>4.83</td>
      <td>4.83</td>
      <td>4.83</td>
      <td>4.67</td>
      <td>4.33</td>
      <td>4.67</td>
      <td>40.0</td>
      <td>12</td>
      <td>20.0</td>
      <td>3.713572</td>
      <td>3.044522</td>
      <td>6</td>
    </tr>
    <tr>
      <th>31785</th>
      <td>742712860134731702</td>
      <td>113295877</td>
      <td>3rd fl. loft style bedroom in Family friendly ...</td>
      <td>Take it easy at this unique and tranquil getaw...</td>
      <td>1</td>
      <td>Tompkinsville</td>
      <td>40.63636</td>
      <td>-74.08908</td>
      <td>Private room in rental unit</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Coffee maker", "Smoke alarm", "Free washer \...</td>
      <td>30</td>
      <td>4.67</td>
      <td>4.83</td>
      <td>4.83</td>
      <td>4.83</td>
      <td>4.67</td>
      <td>4.33</td>
      <td>4.67</td>
      <td>40.0</td>
      <td>1</td>
      <td>20.0</td>
      <td>3.713572</td>
      <td>3.044522</td>
      <td>6</td>
    </tr>
    <tr>
      <th>43942</th>
      <td>742712860134731702</td>
      <td>113295877</td>
      <td>3rd fl. loft style bedroom in Family friendly ...</td>
      <td>Take it easy at this unique and tranquil getaw...</td>
      <td>1</td>
      <td>Tompkinsville</td>
      <td>40.63636</td>
      <td>-74.08908</td>
      <td>Private room in rental unit</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Hot water", "Private backyard \u2013 Fully f...</td>
      <td>30</td>
      <td>4.67</td>
      <td>4.83</td>
      <td>4.83</td>
      <td>4.83</td>
      <td>4.67</td>
      <td>4.33</td>
      <td>4.67</td>
      <td>40.0</td>
      <td>2</td>
      <td>20.0</td>
      <td>3.713572</td>
      <td>3.044522</td>
      <td>6</td>
    </tr>
    <tr>
      <th>56132</th>
      <td>742712860134731702</td>
      <td>113295877</td>
      <td>3rd fl. loft style bedroom in Family friendly ...</td>
      <td>Take it easy at this unique and tranquil getaw...</td>
      <td>1</td>
      <td>Tompkinsville</td>
      <td>40.63636</td>
      <td>-74.08908</td>
      <td>Private room in rental unit</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Free washer \u2013 In unit", "Luggage dropof...</td>
      <td>30</td>
      <td>4.67</td>
      <td>4.83</td>
      <td>4.83</td>
      <td>4.83</td>
      <td>4.67</td>
      <td>4.33</td>
      <td>4.67</td>
      <td>40.0</td>
      <td>3</td>
      <td>20.0</td>
      <td>3.713572</td>
      <td>3.044522</td>
      <td>6</td>
    </tr>
    <tr>
      <th>68427</th>
      <td>742712860134731702</td>
      <td>113295877</td>
      <td>3rd fl. loft style bedroom in Family friendly ...</td>
      <td>Take it easy at this unique and tranquil getaw...</td>
      <td>1</td>
      <td>Tompkinsville</td>
      <td>40.63636</td>
      <td>-74.08908</td>
      <td>Private room in rental unit</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Microwave", "Free street parking", "Extra pi...</td>
      <td>30</td>
      <td>4.67</td>
      <td>4.83</td>
      <td>4.83</td>
      <td>4.83</td>
      <td>4.67</td>
      <td>4.33</td>
      <td>4.67</td>
      <td>40.0</td>
      <td>4</td>
      <td>20.0</td>
      <td>3.713572</td>
      <td>3.044522</td>
      <td>6</td>
    </tr>
    <tr>
      <th>80826</th>
      <td>742712860134731702</td>
      <td>113295877</td>
      <td>3rd fl. loft style bedroom in Family friendly ...</td>
      <td>Take it easy at this unique and tranquil getaw...</td>
      <td>1</td>
      <td>Tompkinsville</td>
      <td>40.63636</td>
      <td>-74.08908</td>
      <td>Private room in rental unit</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Lock on bedroom door", "Free street parking"...</td>
      <td>30</td>
      <td>4.67</td>
      <td>4.83</td>
      <td>4.83</td>
      <td>4.83</td>
      <td>4.67</td>
      <td>4.33</td>
      <td>4.67</td>
      <td>40.0</td>
      <td>5</td>
      <td>20.0</td>
      <td>3.713572</td>
      <td>3.044522</td>
      <td>6</td>
    </tr>
    <tr>
      <th>93293</th>
      <td>742712860134731702</td>
      <td>113295877</td>
      <td>3rd fl. loft style bedroom in Family friendly ...</td>
      <td>Take it easy at this unique and tranquil getaw...</td>
      <td>1</td>
      <td>Tompkinsville</td>
      <td>40.63636</td>
      <td>-74.08908</td>
      <td>Private room in rental unit</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Outdoor furniture", "Mini fridge", "Essentia...</td>
      <td>30</td>
      <td>4.67</td>
      <td>4.83</td>
      <td>4.83</td>
      <td>4.83</td>
      <td>4.67</td>
      <td>4.33</td>
      <td>4.67</td>
      <td>40.0</td>
      <td>6</td>
      <td>20.0</td>
      <td>3.713572</td>
      <td>3.044522</td>
      <td>6</td>
    </tr>
    <tr>
      <th>106090</th>
      <td>742712860134731702</td>
      <td>113295877</td>
      <td>3rd fl. loft style bedroom in Family friendly ...</td>
      <td>Take it easy at this unique and tranquil getaw...</td>
      <td>1</td>
      <td>Tompkinsville</td>
      <td>40.63636</td>
      <td>-74.08908</td>
      <td>Private room in rental unit</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Free washer \u2013 In unit", "Lock on bedroo...</td>
      <td>30</td>
      <td>4.67</td>
      <td>4.83</td>
      <td>4.83</td>
      <td>4.83</td>
      <td>4.67</td>
      <td>4.33</td>
      <td>4.67</td>
      <td>40.0</td>
      <td>7</td>
      <td>20.0</td>
      <td>3.713572</td>
      <td>3.044522</td>
      <td>6</td>
    </tr>
    <tr>
      <th>118786</th>
      <td>742712860134731702</td>
      <td>113295877</td>
      <td>3rd fl. loft style bedroom in Family friendly ...</td>
      <td>Take it easy at this unique and tranquil getaw...</td>
      <td>1</td>
      <td>Tompkinsville</td>
      <td>40.63636</td>
      <td>-74.08908</td>
      <td>Private room in rental unit</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Outdoor furniture", "Free washer \u2013 In u...</td>
      <td>30</td>
      <td>4.67</td>
      <td>4.83</td>
      <td>4.83</td>
      <td>4.83</td>
      <td>4.67</td>
      <td>4.33</td>
      <td>4.67</td>
      <td>40.0</td>
      <td>8</td>
      <td>20.0</td>
      <td>3.713572</td>
      <td>3.044522</td>
      <td>6</td>
    </tr>
    <tr>
      <th>131491</th>
      <td>742712860134731702</td>
      <td>113295877</td>
      <td>3rd fl. loft style bedroom in Family friendly ...</td>
      <td>Take it easy at this unique and tranquil getaw...</td>
      <td>1</td>
      <td>Tompkinsville</td>
      <td>40.63636</td>
      <td>-74.08908</td>
      <td>Private room in rental unit</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Toaster", "Wifi", "Host greets you", "Mini f...</td>
      <td>30</td>
      <td>4.67</td>
      <td>4.83</td>
      <td>4.83</td>
      <td>4.83</td>
      <td>4.67</td>
      <td>4.33</td>
      <td>4.67</td>
      <td>40.0</td>
      <td>9</td>
      <td>20.0</td>
      <td>3.713572</td>
      <td>3.044522</td>
      <td>6</td>
    </tr>
    <tr>
      <th>144366</th>
      <td>742712860134731702</td>
      <td>113295877</td>
      <td>3rd fl. loft style bedroom in Family friendly ...</td>
      <td>Take it easy at this unique and tranquil getaw...</td>
      <td>1</td>
      <td>Tompkinsville</td>
      <td>40.63636</td>
      <td>-74.08908</td>
      <td>Private room in rental unit</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Mini fridge", "Extra pillows and blankets", ...</td>
      <td>30</td>
      <td>4.67</td>
      <td>4.83</td>
      <td>4.83</td>
      <td>4.83</td>
      <td>4.67</td>
      <td>4.33</td>
      <td>4.67</td>
      <td>40.0</td>
      <td>10</td>
      <td>20.0</td>
      <td>3.713572</td>
      <td>3.044522</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>


    
    Sample data for listing ID 44251682:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>host_id</th>
      <th>name</th>
      <th>description</th>
      <th>host_is_superhost</th>
      <th>neighbourhood_cleansed</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>property_type</th>
      <th>room_type</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>amenities</th>
      <th>minimum_nights</th>
      <th>review_scores_rating</th>
      <th>review_scores_accuracy</th>
      <th>review_scores_cleanliness</th>
      <th>review_scores_checkin</th>
      <th>review_scores_communication</th>
      <th>review_scores_location</th>
      <th>review_scores_value</th>
      <th>price</th>
      <th>month</th>
      <th>price_per_person</th>
      <th>log_price</th>
      <th>log_price_per_person</th>
      <th>total_reviews</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>89693</th>
      <td>44251682</td>
      <td>356623718</td>
      <td>Cozy Room w/bathroom perfect for medical staffs</td>
      <td>Bright and clean room perfect for long stays. ...</td>
      <td>1</td>
      <td>Stapleton</td>
      <td>40.62667</td>
      <td>-74.0798</td>
      <td>Private room in home</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Essentials", "Cooking basics", "Stove", "Dis...</td>
      <td>30</td>
      <td>4.6</td>
      <td>4.6</td>
      <td>5.0</td>
      <td>4.6</td>
      <td>4.6</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>75.0</td>
      <td>6</td>
      <td>37.5</td>
      <td>4.330733</td>
      <td>3.650658</td>
      <td>10</td>
    </tr>
    <tr>
      <th>102522</th>
      <td>44251682</td>
      <td>356623718</td>
      <td>Cozy Room w/bathroom perfect for medical staffs</td>
      <td>Bright and clean room perfect for long stays. ...</td>
      <td>1</td>
      <td>Stapleton</td>
      <td>40.62667</td>
      <td>-74.0798</td>
      <td>Private room in home</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Free parking on premises", "Dining table", "...</td>
      <td>30</td>
      <td>4.6</td>
      <td>4.6</td>
      <td>5.0</td>
      <td>4.6</td>
      <td>4.6</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>75.0</td>
      <td>7</td>
      <td>37.5</td>
      <td>4.330733</td>
      <td>3.650658</td>
      <td>10</td>
    </tr>
    <tr>
      <th>115251</th>
      <td>44251682</td>
      <td>356623718</td>
      <td>Cozy Room w/bathroom perfect for medical staffs</td>
      <td>Bright and clean room perfect for long stays. ...</td>
      <td>1</td>
      <td>Stapleton</td>
      <td>40.62667</td>
      <td>-74.0798</td>
      <td>Private room in home</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Hot water kettle", "Microwave", "Dining tabl...</td>
      <td>30</td>
      <td>4.6</td>
      <td>4.6</td>
      <td>5.0</td>
      <td>4.6</td>
      <td>4.6</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>75.0</td>
      <td>8</td>
      <td>37.5</td>
      <td>4.330733</td>
      <td>3.650658</td>
      <td>10</td>
    </tr>
    <tr>
      <th>127920</th>
      <td>44251682</td>
      <td>356623718</td>
      <td>Cozy Room w/bathroom perfect for medical staffs</td>
      <td>Bright and clean room perfect for long stays. ...</td>
      <td>1</td>
      <td>Stapleton</td>
      <td>40.62667</td>
      <td>-74.0798</td>
      <td>Private room in home</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Toaster", "Room-darkening shades", "Wifi", "...</td>
      <td>30</td>
      <td>4.6</td>
      <td>4.6</td>
      <td>5.0</td>
      <td>4.6</td>
      <td>4.6</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>75.0</td>
      <td>9</td>
      <td>37.5</td>
      <td>4.330733</td>
      <td>3.650658</td>
      <td>10</td>
    </tr>
    <tr>
      <th>140712</th>
      <td>44251682</td>
      <td>356623718</td>
      <td>Cozy Room w/bathroom perfect for medical staffs</td>
      <td>Bright and clean room perfect for long stays. ...</td>
      <td>1</td>
      <td>Stapleton</td>
      <td>40.62667</td>
      <td>-74.0798</td>
      <td>Private room in home</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>["Clothing storage: closet and dresser", "Room...</td>
      <td>30</td>
      <td>4.6</td>
      <td>4.6</td>
      <td>5.0</td>
      <td>4.6</td>
      <td>4.6</td>
      <td>4.5</td>
      <td>4.5</td>
      <td>75.0</td>
      <td>10</td>
      <td>37.5</td>
      <td>4.330733</td>
      <td>3.650658</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>


### 6. Finalize & Save Modeling Dataset


```python
# Save to Parquet
output_path = os.path.join(OUTPUT_DATA_DIR, OUTPUT_FILENAME)
print(f"\nSaving to {output_path}...")
final_df.to_parquet(output_path, index=False)
print("Done.")
```

    
    Saving to ../data/nyc/nyc_dataset_oct_17.parquet...
    Done.



```python
# Save a sample with all occurrences of 2 random listings
sample_ids = np.random.choice(final_df['id'].unique(), size=2, replace=False)
sample_df = final_df[final_df['id'].isin(sample_ids)]
sample_output_path = os.path.join(OUTPUT_DATA_DIR, f"{CITY}_sample_listings.csv")
sample_df.to_csv(sample_output_path, index=False)
print(f"Sample listings saved to {sample_output_path}.")
```

    Sample listings saved to ../data/nyc/nyc_sample_listings.csv.

