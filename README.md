# Koi Fish Classification

Raising koi fish is a growing hobby in the United States, driven by increasing demand for ornamental pets and the rising popularity of backyard ponds and garden landscaping. The aesthetic appeal and calming presence of koi ponds are drawing more homeowners to purchase their first koi.

In contrast, across the world in Japan, koi keeping is more than a visual hobby--it carries deep cultural significance. Breeding, raising, and trading koi is a refined art form that has evolved over centuries. Many of the most sought-after koi in the U.S. are still those imported from Japan or bred from prestigious Japanese lineages.

For a first-time buyer, whether online or at a local store, the wide variety of patterns, colors, and markings can be overwhelming. Even more surprising is the price tag. While juvenile koi can cost as little as $10, champion-grade koi can sell for tens of thousands of dollars.

Koi fish prices range from $10 for a 5” juvenile to over $50,000 for champion level quality.[(1)](https://www.kodamakoifarm.com/how-much-do-koi-fish-cost/)
The most expensive koi fish ever sold was a Kohaku named "S Legend", which fetched a staggering $1.8 million in 2018. [(2)](https://www.businessinsider.com/koi-fish-worth-millions-expensive-japan-2018-12)

# Project Goal

The goal of this project is to train an Image Classification Model to assist koi hobbyists in identifying koi fish varieties, sub-varieties, and traits. Since these visual and genetic distinctions significantly influence a koi’s market value, the tool aims to empower buyers and sellers to better understand what a fish is worth. May they be making their first purchase or selling a fish that they had raised.

# Table of Contents

1. [About the Data](#About-the-Data)
2. Data Insights
3. Modeling
4. Results
5. Next Steps
6. References

# About the Data

### Data Source 

Data was gathered from six different online koi retailers, and obtained a total of 3,844:
1. [Grand Koi](https://www.grandkoi.com/product-category/other-koi-varieties-for-sale/shiro-utsuri/) -> 367 Observations
2. [Sacramento Koi](https://sacramentokoi.com/koi/?orderby=popularity&hide_sold_products=true) -> 423 Observations
3. [Kloubec Koi](https://www.kloubeckoi.com/koi-fish-for-sale/) -> 517 Observations
4. [NextDayKoi](https://nextdaykoi.com/shop/) -> 1948 Observations
5. [GCKoiFarm](https://gckoi.com/collections/koi) -> 408 Observations
6. [Champkoi](https://www.champkoi.com/collections/all-koi) -> 185 Observations

### Data Quality

Each online retailer provided different levels of detail for each fish listing.
![Screenshot 2025-05-22 at 12 00 33 PM](https://github.com/user-attachments/assets/e0e39595-d131-4b82-ab78-578e81728379)

For classification purposes, the most important elements are the image of the fish and the tags that indicate its variety and any applicable traits.

To ensure clean and relevant input for modeling, the images were carefully filtered. Listings with multiple fish in one image were excluded to avoid confusion during training. Since the model focuses on distinguishing between koi fish varieties, images containing non-koi fish (e.g., goldfish) or non-fish content were also removed.

![Screenshot 2025-05-22 at 12 06 58 PM](https://github.com/user-attachments/assets/5ce32b20-61d8-4c5c-b93b-763489f7a1d8)

After cleaning, the dataset was reduced from 3,844 to 3,134 observations. The remaining entries yielded the following tag frequency distribution.

Note that a single fish can have multiple tags, and some tags represent broader classification categories. For example, the tag "Gosanke" encompasses three specific varieties: Kohaku, Sanke, and Showa.

![tags_frequency](https://github.com/user-attachments/assets/8f196ea6-8137-425b-8321-79c935affe94)

### Selected Dataset

My model will focus on distinguishing between 3 varieties and the presence of 2 traits. The varieties include: Kohaku, Sanke and Showa. The traits are GinRin and Tancho. 

![Screenshot 2025-05-22 at 12 39 55 PM](https://github.com/user-attachments/assets/808d97fc-9d8f-421b-8a95-1e145b31c830)

The selected dataset has 5 labels (3 varities and 2 traits). Most fish have only 1 label, some have 2 and very few have 3 labels, as shown in the plot below.

![num_labels_per_sample_hist](https://github.com/user-attachments/assets/da56c122-2f7f-43a7-b24e-438a2353a6b0)




