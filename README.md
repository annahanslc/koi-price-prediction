# 🎏 Koi Fish Classification

![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-96%25-green)


Raising koi fish is a growing hobby in the United States, driven by increasing demand for ornamental pets and the rising popularity of backyard ponds and garden landscaping. The aesthetic appeal and calming presence of koi ponds are drawing more homeowners to purchase their first koi.

For a first-time buyer, whether online or at a local store, the wide variety of patterns, colors, and markings can be overwhelming. Even more surprising is the price tag 💸. 

Koi fish prices range from $10 for a 5” juvenile to over $50,000 for champion level quality.[(1)](https://www.kodamakoifarm.com/how-much-do-koi-fish-cost/)
The most expensive koi fish ever sold was a Kohaku named "S Legend", which fetched a staggering $1.8 million in 2018. [(2)](https://www.businessinsider.com/koi-fish-worth-millions-expensive-japan-2018-12)

# Project Goal

The goal of this project is to train an Image Classification Model to assist koi hobbyists in identifying koi fish varieties, sub-varieties, and traits. Since these visual and genetic distinctions significantly influence a koi’s market value, the tool aims to empower buyers and sellers to better understand what a fish is worth. May they be making their first purchase or selling a fish that they had raised.

# Table of Contents

1. [About the Data](#About-the-Data)
2. [Modeling](#Modeling)
3. [Results](#Results)
4. [Next Steps](#Next-Steps)
5. [References](#References)

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

After cleaning, the dataset was reduced from 3,844 to **3,134 observations**. The remaining entries yielded the following tag frequency distribution.

Note that a single fish can have multiple tags, and some tags represent broader classification categories. For example, the tag "Gosanke" encompasses three specific varieties: Kohaku, Sanke, and Showa.

<img src='https://github.com/user-attachments/assets/8f196ea6-8137-425b-8321-79c935affe94' width=800 />

### Subset

My model will focus on distinguishing between 3 varieties and the presence of 2 traits. The varieties include: Kohaku, Sanke and Showa. The traits are GinRin and Tancho. 

<img src='https://github.com/user-attachments/assets/808d97fc-9d8f-421b-8a95-1e145b31c830' width=800 />

The selected dataset has **5 labels** (3 varities and 2 traits) with **859 observations**. Since the varieties are mutually-exclusive, a fish should not have more than 1 variety, but can have multiple traits, so the number of labels per fish should be between 1 and 3. Most fish have only 1 label, some have 2 and very few have 3 labels, as shown in the plot below. 

<img src='https://github.com/user-attachments/assets/da56c122-2f7f-43a7-b24e-438a2353a6b0' width=500 />

Random sample of each label. The below show a random selection of 20 samples from each label.

### Kohaku
<img src='https://github.com/user-attachments/assets/e2004c8d-acd0-471a-a71b-96f7349ab6b7' width=800 />

### Sanke
<img src='https://github.com/user-attachments/assets/869d16e0-26fc-4520-97a2-2bff107e117a' width=800 />

### Showa
<img src='https://github.com/user-attachments/assets/2d68d20a-5dcf-4298-9171-0e58d425a41a' width=800 />

### Tancho
<img src='https://github.com/user-attachments/assets/bb72ef3a-985f-42f1-a2a5-4ec0902d5681' width=800 />

### GinRin
<img src='https://github.com/user-attachments/assets/7c339a6c-4a9e-43c5-819e-88ea26590656' width=800 />

### Class Balance

The number of samples per label is **imbalanced**, with Tancho being the most underrepresented, having only 20–25% as many observations as the other labels. To address this during modeling, two strategies were applied:

1. Stratified data splitting to preserve label distribution across training, validation, and test sets.
2. Class weighting, using positive weights to give more importance to underrepresented classes during training.

<img src='https://github.com/user-attachments/assets/0ab0f502-4968-452c-9602-ee1d246af4d3' width=500 />


# Modeling

My best model uses transfer learning with a **ResNet18** backbone for multi-label image classification. The final fully connected (FC) layer is replaced with the outputs corresponding to the shape of the koi classification. The ResNet18 weights unfrozen yielded better results than frozen weights as it to allowed for more comprehensive learning. 

I used **nn.BCEWithLogitsLoss** as the loss function because this is a multi-label classification problem, where each sample can belong to multiple classes simultaneously. BCEWithLogitsLoss is a numerically stable combination of Sigmoid activation and Binary Cross-Entropy loss. Instead of applying sigmoid separately followed by BCELoss, it integrates both steps into a single function, improving computational stability and training efficiency—especially when dealing with logits that have extreme values.

The model uses the Adam optimizer with an inital learning rate of 0.001, for fast convergence, but then the learning rate is controlled by the ReduceLROnPlateau scheduler, which monitors the validation loss and reduced the learning rate by a factor of 0.1 if the loss does not improve for 3 consecutive epochs. This helps the model fine-tune during the later stages of training.

The training loop was set for 200 epochs, but with early stopping with a patience of 10, so if the validation loss does not improve for 10 epochs in a row, the training loop will exit immediately. 

# Results

For the calculate of accuracy, the model's prediction is correct only if it predicted all the labels for an observation correctly. 
For example, if the fish is Kohaku and GinRin, but the model only predicted Kohaku, this would be considered as an incorrect prediction. 

The best model yielded **92.86%** accuracy on the validation dataset, and **96.97%** accuracy on the test (holdout) dataset. 

![Screenshot 2025-05-22 at 1 22 44 PM](https://github.com/user-attachments/assets/33e9c753-2cf8-4eea-ab6e-abaae2281b09)

### Val Results

![Screenshot 2025-05-22 at 1 25 13 PM](https://github.com/user-attachments/assets/2cfa3cf6-23d0-4a88-903b-64056849843d)

<img src='https://github.com/user-attachments/assets/8b47d1d4-51e0-40c4-b046-a770badb0bfd' width=800 />

In the validation data, the model made the highest number of errors for the Ginrin (sparkly scales) label in the form of false negatives: failing to detect the presence of the trait when it was actually there.

Upon manual inspection, many of these images were challenging to interpret even for the human eye. The shininess of Ginrin scales is angle-dependent, and inconsistent lighting conditions can easily obscure this trait. In some photos, the metallic sparkle was faint or missing altogether, making it nearly impossible to confidently confirm the trait.

This highlights an element of irreducible error in the dataset: no matter how accurate the model, it cannot detect visual cues that are not clearly captured in the image. When trait visibility depends heavily on photography quality, such as lighting, resolution, and angle, even the best model is limited by the input it receives.


### Test Results

![Screenshot 2025-05-22 at 1 27 13 PM](https://github.com/user-attachments/assets/5934465e-92e9-47d0-8a31-0147b89783ba)

<img src='https://github.com/user-attachments/assets/105d3e16-abc7-4cd5-b1b2-cbc2de56b663' width=800 />

On the test dataset, the model only made 4 incorrect predictions:
1. False positive for Kohaku
2. False negative for Sanke
3. False positive for Tancho
4. False positive for GinRin
   

### Checking the Wrong Predictions

![mistakes](https://github.com/user-attachments/assets/96079acb-2be3-4d91-a16a-d16f429e974c)

Far Left: The model correctly identified the Tancho trait but failed to detect Sanke. Sanke koi are typically recognized by their distinctive black markings, which are visible in the image. The model’s omission of the Sanke label represents a missed classification despite clear visual cues.

Second from the Left: The model accurately predicted the Kohaku variety but missed the Ginrin (sparkly scales) trait. Even to the human eye, the sparkle effect is subtle and difficult to discern in this image. This highlights a key source of irreducible error, as the visibility of the Ginrin trait is highly dependent on image quality and lighting angle.

Second from the Right: The model correctly detected Ginrin but falsely labeled the fish as Tancho. A manual review clearly shows that the fish lacks the characteristic red cap associated with the Tancho trait. This is a definitive model error.

Far Right: The model predicted Kohaku and Ginrin, while the actual fish is Ginrin only. However, this fish is an Aigoromo, a variety that features indigo netting over the red markings of a Kohaku pattern. Since the model was not trained to distinguish Aigoromo specifically, and the underlying pattern resembles Kohaku, the prediction is reasonable and justifiable given the training constraints.
   
# Next Steps

As a next step, I plan to extend this project by incorporating a price regression model. The goal is to predict the estimated market price of a koi fish based on a combination of:

The labels predicted by the current multi-label classification model (e.g., variety, traits like Ginrin or Tancho),

And additional structured information about the fish, such as:
1. Size (length in inches),
2. Age (juvenile vs. adult),
3. Sex (if known),
4. And potentially breeder or origin (e.g., Japanese vs. domestic).

This regression model would allow hobbyists and sellers to obtain a data-driven price estimate, which could serve as a baseline reference for buying, selling, or evaluating koi fish. Similar to how Zillow provides estimated home values with its "Zestimate," this model aims to bring transparency and consistency to the often subjective and opaque world of koi pricing.

Ultimately, combining image-based classification with feature-driven price estimation will create a more complete tool for both newcomers and seasoned collectors in the koi community.

# References

1. **Kodama Koi Farm.** _How Much Do Koi Fish Cost?_  https://www.kodamakoifarm.com/how-much-do-koi-fish-cost/
2. **Business Insider.** _This Japanese Koi Fish Sold for $1.8 Million — Here's Why It's So Expensive._ https://www.businessinsider.com/koi-fish-worth-millions-expensive-japan-2018-12 
3. **Grand Koi.** _Shiro Utsuri and Other Koi Varieties for Sale._  https://www.grandkoi.com/product-category/other-koi-varieties-for-sale/shiro-utsuri/
4. **Sacramento Koi.** _Koi Fish for Sale._  https://sacramentokoi.com/koi/?orderby=popularity&hide_sold_products=true
5. **Kloubec Koi Farm.** _Koi Fish for Sale._  https://www.kloubeckoi.com/koi-fish-for-sale/
6. **Next Day Koi.** _Shop All Koi Fish._  https://nextdaykoi.com/shop/
7. **GC Koi Farm.** _Koi Collection._  https://gckoi.com/collections/koi
8. **Champ Koi.** _All Koi for Sale._  https://www.champkoi.com/collections/all-koi

