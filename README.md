# **ML-DS-Practice**

## **Data Description:**
### **Original data source (Kaggle):** 
https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data
### **All original data plus additional files used in the repository (Google Drive):**
https://drive.google.com/drive/folders/1OcNdg6yC0V4evJfgyLjD8lPho5fnxpg4?usp=drive_link

**All files list:**
- **sales_train.csv** - the training set. Daily historical data from January 2013 to October 2015.
- **test.csv** - the test set. You need to forecast the sales for these shops and products for November 2015.
- **sample_submission.csv** - a sample submission file in the correct format.
- **items.csv** - supplemental information about the items/products.
- **item_categories.csv**  - supplemental information about the items categories.
- **shops.csv** - supplemental information about the shops.
- **merged_train.csv**, **merged_test.csv** and **merged_train_aggregated.csv** - all original files merged and cleared
- **full_train.csv** - contains **"item_cnt_month"** value for each ("date_block_num", "shop_id") pair; used in EDA and optionally in Feature Extraction instead of 'merged_train_aggregated.csv'
- **result_test.csv** and *result_train.csv* - datasets with all extracted features
- **result_test_full.csv** and **result_train_full.csv** - datasets based on 'full_train.csv' with all extracted features
- **models/** folder with some models saved (used only in 'modeling.ipynb' notebook)