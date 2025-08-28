Wrangling (also called _data preprocessing_) means **cleaning and preparing raw data** so it can be used effectively in ML models.

## Cleaning
---
- **Remove or fix errors** (missing values, duplicates, wrong formats).
- Makes data reliable.

**Real-World Example**  
A hospital’s patient records may have missing ages or duplicate entries. Cleaning ensures only valid patient info is used for analysis.

## Normalization
---
- **Scale values** into a similar range so no feature dominates.
- Important for algorithms like gradient descent and K-means.

**Real-World Example**  
In banking data: _Age_ (20–70) vs _Income_ (20,000–200,000). Normalization ensures both matter equally in predicting loan approval.

## Train/Test Split
---
- **Divide data** into training (to teach the model) and testing (to check accuracy).
- Prevents overfitting.

**Real-World Example**  
In a spam filter:
- Training set → teaches the model what spam looks like.
- Test set → checks if the model correctly classifies new emails.