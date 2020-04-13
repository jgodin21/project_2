# Project 2: Predicting Sale Prices in Ames Housing Dataset

## Data Science Problem

We have been tasked with creating a regression model to predict the price of a house at sale from the Ames Housing Dataset, under the guise of taking the model to a real estate company and selling them on the model's merits.

**What influences home purchase prices, and which features are needed to minimize out-of-sample prediction error?** What should homeowners improve to increase the value of their homes? Which features might not be worth investing in? If buyers are looking for certain features, how much should they expect to pay?

## Executive Summary

For this project, I utilized the Ames Housing Dataset, which contains home sales data between 2006 and 2010 from the municipality of Ames, IA. Two data files were provided, a training file used for building the regression model consisting of 2,051 records, and a separate, external test file used for validating the model's performance; the test data included 878 additional records.

In total, the Ames dataset includes 78 different features that describe a home and the land it occupies. The final column, SalePrice, only exists in the training file, and that is what we seek to predict in the test data file.

The two files were cleaned extensively in sequence, starting with the training data. Whatever cleaning was done on the training file was then repeated on the test file, synching any data issues that arose from dataset to dataset, such as dropping feature levels that did not exist in one or the other file.

In addition to cleaning the file for missing or incorrect data, any categorical variables were converted to 0/1 dummy variables. Additional feature engineering included creating interaction effects and non-linear terms (squared and or cubed values of features), log-transformation of variables (including SalePrice) with large magnitudes, as well as recasting certain outlier data with additional threshold dummy variables. **The final candidate set of predictor variables totaled 179**!

I ended up attempting **four different approaches** to the modeling, all with the log-transformed sale price as the dependent variable. These approaches included:
- **Standard linear regression** with a mixture of "raw" and log-transformed predictors, trying to find a relatively simple model to explain SalePrice
- **PowerTransformed regression** models where all variables in the model get transformed prior to model estimation using scikit-learn's PowerTransformer module
- **Ensemble regression modeling**, where I estimated six different PowerTransformed regression models, each including a core set of key predictors plus thematic subsets of additional predictors unique to each sub-model. The final predictions coming out of this approach were the average predicted sale prices across the six models.
- **LASSO regression model** - this model resulted in the **final, best predictions with the lowest amount of error**. Here, all 179 candidate predictors were included, with the LASSO regularization ultimately selecting which features contribute to improving out-of-sample prediction accuracy, setting non-impactful features to zero. **The final model retained 107 of these features, with a testing-subsample R-square of 0.8999**.

In order to understand the contribution of such a large set of predictors to sale price, I utilized the SHAP Python package to estimate Shapley Values from the model. Shapley Values arose out of Game Theory, where the goal was to quantify how much each player contributed to a victory.

In addition to creating a final presentation based on the modeling, we entered our models in a GA Data Science 11 cohort-wide private kaggle competition, with final rankings based on an additional holdout set of test data. My model finished in 8th place overall (out of 89 entries), 1st in the Boston office, with a final root mean squared error of $19,155.

## Data Dictionary

**Data Sources**

- A full explanation of the data file can be found on [kaggle.com](https://www.kaggle.com/c/dsi-us-11-project-2-regression-challenge/data)

|Feature|Type|Treatment|<p align="left">Description</p>|
|---|---|---|---|
|**SalePrice**|float|Log-transformed|<p align="left">The property's sale price in dollars, the **target variable** I was trying to predict for this challenge. Only included in Training data file.</p>|
|MSSubClass| object |Dummy-coded|<p align="left">The building class, consisting of 16 levels, such as 1-Story 1946 & Newer All Styles, Split or Multi-Level, Duplex, etc.</p>|
|MSZoning|object|Dummy-coded|<p align="left">Identifies the general zoning classification of the sale, such as Residential High or Low Density. 8 total levels.</p>|
|LotFrontage|float|Missing values predicted by regression sub-model|<p align="left">Linear feet of street connected to property.</p>|
|LotArea|float|Log-transformed, plus indicator for very large lots|<p align="left">Lot size in square feet.</p>|
|Street|object|Dropped from model due to little-to-no variation in data|<p align="left">Type of road access to property (gravel or paved).</p>|
|Alley|object|Dropped from model due to little-to-no variation in data|<p align="left">Type of alley access to property. 3 levels.</p>|
|LotShape|object|Dummy-coded|<p align="left">General shape of the property. 4 levels, Regular to Irregular.</p>|
|LandContour|object|Dummy-coded|<p align="left">Flatness of the property. 4 levels.</p>|
|Utilities|object|Dummy-coded|<p align="left">Type of utilities available. 4 levels.</p>|
|LotConfig|object|Dummy-coded|<p align="left">Lot configuration. 5 levels.</p>|
|LandSlope|object|Dummy-coded|<p align="left">Slope of property. 3 levels.</p>|
|Neighborhood|object|Dummy-coded|<p align="left">Physical locations within Ames city limits. 25 levels.</p>|
|Condition1|object|Dummy-coded|<p align="left">Proximity to main road or railroad. 9 levels. </p>|
|Condition2|object|Dummy-coded|<p align="left">Proximity to main road or railroad (if a second is present). 9 levels.</p>|
|BldgType|object|Dummy-coded|<p align="left">Type of dwelling. 5 levels.</p>|
|HouseStyle|object|Dummy-coded|<p align="left">Style of dwelling. 8 levels.</p>|
|OverallQual|int|As-is|<p align="left">Overall material and finish quality. 1 to 10 scale from Very Poor to Very Excellent.</p>|
|OverallCond|int|As-is|<p align="left">Overall condition rating. 1 to 10 scale from Very Poor to Very Excellent.</p>|
|YearBuilt|int|Subtracted from YrSold to create Age At Sale variable|<p align="left">Original construction date.</p>|
|YearRemodAdd|int|Subtracted from YrSold to create Years Since Remod/Add variable|<p align="left">Remodel date (same as construction date if no remodeling or additions. Data only goes back to 1950.)</p>|
|RoofStyle|object|Converted to IsHipRoof dummy variable|<p align="left">Type of roof. 6 levels.</p>|
|RoofMatl|object|Dummy-coded|<p align="left">Roof material. 8 levels.</p>|
|Exterior1st|object|Dummy-coded|<p align="left">Exterior covering on house. 17 levels.</p>|
|Exterior2nd|object|Converted to DiffExt2 dummy variable if different from Exterior 1st|<p align="left">Exterior covering on house (if more than one material). 17 levels.</p>|
|MasVnrType|object|Dummy-coded|<p align="left">Masonry veneer type. 5 levels.</p>|
|MasVnrArea|float|Dropped|<p align="left">Masonry veneer area in square feet.</p>|
|ExterQual|object|Dummy-coded|<p align="left">Exterior material quality. 5 levels, from Poor to Excellent.</p>|
|ExterCond|object|Dummy-coded|<p align="left">Present condition of the material on the exterior. 5 levels from Poor to Excellent.</p>|
|Foundation|object|Dummy-coded|<p align="left">Type of foundation. 6 levels.</p>|
|BsmtQual|object|Converted to HasExGdBsmtHeight dummy|<p align="left">Height of the basement, 6 levels from No Basement to Excellent (100+ inches).</p>|
|BsmtCond|object|Dummy-coded|<p align="left">General condition of the basement. 6 levels, from No Basement to Excellent.</p>|
|BsmtExposure|object|Dummy-coded|<p align="left">Walkout or garden level basement walls. 5 levels, from No Basement to Good Exposure.</p>|
|BsmtFinType1|object|Dummy-coded|<p align="left">Quality of basement finished area. 7 levels from No Basement to Good Living Quarters.</p>|
|BsmtFinSF1|float|As-is|<p align="left">Type 1 finished square feet.</p>|
|BsmtFinType2|object|Dummy-coded|<p align="left">Quality of second finished area (if present). 7 levels from No Basement to Good Living Quarters.</p>|
|BsmtFinSF2|float|Dropped|<p align="left">Type 2 finished square feet.</p>|
|BsmtUnfSF|float|Dropped|<p align="left">Unfinished square feet of basement area.</p>|
|TotalBsmtSF|float|As-is|<p align="left">Total square feet of basement area.</p>|
|Heating|object|Dummy-coded|<p align="left">Type of heating. 6 levels.</p>|
|HeatingQC|object|Dummy-coded|<p align="left">Heating quality and condition. 5 levels, from Poor to Excellent.</p>|
|CentralAir|object|Dummy-coded|<p align="left">Central air conditioning. Present/absent.</p>|
|Electrical|object|Converted to NotStdElectrical dummy variable|<p align="left">Electrical system. 5 levels from Mixed to Standard Circuit Breakers.</p>|
|1stFlrSF|float|Log-transformed with additional indicator for very large Values|<p align="left">First floor square feet</p>|
|2ndFlrSF|float|Log-transformed|<p align="left">Second floor square feet.</p>|
|LowQualFinSF|float|As-is|<p align="left">Low quality finished square feet (all floors).</p>|
|GrLivArea|float|Log-transformed|<p align="left">Above grade (ground) living area square feet.</p>|
|BsmtFullBath|int|Converted to HasBsmtFullBath dummy variable|<p align="left">Basement full bathrooms.</p>|
|BsmtHalfBath|int|Converted to HasBsmtHalfBath dummy variable|<p align="left">Basement half bathrooms.</p>|
|FullBath|int|As-is|<p align="left">Full bathrooms above grade.</p>|
|HalfBath|int|As-is|<p align="left">Half baths above grade.</p>|
|Bedroom|int|As-is|<p align="left">Number of bedrooms above basement level.</p>|
|KitchenAbvGr|int|As-is|<p align="left">Number of kitchens.</p>|
|KitchenQual|object|Dummy-coded|<p align="left">Kitchen quality. 5 levels, from Poor to Excellent.</p>|
|TotRmsAbvGrd|int|As-is|<p align="left">Total rooms above grade (does not include bathrooms).</p>|
|Functional|object|Converted into IsNotTypFunctional dummy variable|<p align="left">Home functionality rating. 8 levels, from Salvage Only to Typical.</p>|
|Fireplaces|int|Converted to HasFireplace dummy variable|<p align="left">Number of fireplaces.</p>|
|FireplaceQu|object|Converted to HasExGdFireplace dummy variable|<p align="left">Fireplace quality. 6 levels, from No Fireplace to Excellent.</p>|
|GarageType|object|Dummy-coded|<p align="left">Garage location. 7 levels.</p>|
|GarageYrBlt|int|Converted to GarageBuiltAfterHouse dummy variable|<p align="left">Year garage was built.</p>|
|GarageFinish|object|Dummy-coded|<p align="left">Interior finish of the garage. 4 levels, No Garage to Finished.</p>|
|GarageCars|int|As-is|<p align="left">Size of garage in car capacity.</p>|
|GarageArea|float|As-is|<p align="left">Size of garage in square feet.</p>|
|GarageQual|object|Dropped|<p align="left">Garage quality. 6 levels, from No Garage to Excellent.</p>|
|GarageCond|object|Dropped|<p align="left">Garage condition. 6 levels, from No Garage to Excellent.</p>|
|PavedDrive|object|Converted to IsPavedDrive dummy variable|<p align="left">Paved Driveway (yes, partial, no).</p>|
|WoodDeckSF|float|As-is|<p align="left">Wood deck area in square feet.</p>|
|OpenPorchSF|float|As-is|<p align="left">Open porch area in square feet.</p>|
|EnclosedPorch|float|As-is|<p align="left">Enclosed porch area in square feet.</p>|
|3SsnPorch|float|As-is|<p align="left">Three season porch area in square feet.</p>|
|ScreenPorch|float|As-is|<p align="left">Screen porch area in square feet.</p>|
|PoolArea|float|As-is|<p align="left">Pool area in square feet.</p>|
|PoolQC|object|Dropped due to low incidence of pools in dataset|<p align="left">Pool quality. 5 levels, from No Pool to Excellent.</p>|
|Fence|object|Converted to HasFence dummy variable|<p align="left">Fence quality. 5 levels, from No Fence to Good Privacy.</p>|
|MiscFeature|object|Dropped due to low frequency counts|<p align="left">Miscellaneous feature not covered in other categories. 6 levels.</p>|
|MiscVal|int|As-is|<p align="left">Value of miscellaneous feature.</p>|
|MoSold|int|As-is|<p align="left">Month sold (numeric value).</p>|
|YrSold|int|Used only as difference between year built or year remodeled|<p align="left">Year sold.</p>|
|SaleType|object|Dummy-coded|<p align="left">Type of sale. 10 levels.</p>|

## Contents
**Code**
- [01 Project Overview](./code/01%20project%20overview.ipynb)
- [02 Data Cleaning Train Data](./code/02%20data%20cleaning%20train%20data.ipynb)
- [03 Data Cleaning Test Data](./code/03%20data%20cleaning%20test%20data.ipynb)
- [04 Exploratory Data Analysis](./code/04%20exploratory%20data%20analysis.ipynb)
- [05 Standard Regression Models](./code/05%20standard%20models.ipynb)
- [06 PowerTransformer Models](./code/06%20powertransformer%20models.ipynb)
- [07 Ensemble Models](./code/07%20ensemble%20modeling.ipynb)
- [08 Lasso Model](./code/08%20lasso%20models.ipynb)

**Presentation**
- [09 Ames Housing Regression Modeling Report](./presentation/Ames%20Housing%20-%20Jon%20Godin%20v01.pdf)


## Conclusions and Recommendations

The LASSO regression model is an excellent approach when trying to achieve high accuracy/low model error in the presence of a large number of predictors. It also serves a dual purpose, allowing us to diagnose what drives home price value.

**Key drivers of sale price include:**:
- Gross Living Area & 1st Floor Square Footage
- Overall Quality & Condition ratings of the house
- Age At Sale & Years Since Last Remodel/Addition
- Lot Area

**Additional features with high value include**:
- Excellent Kitchen Quality
- Finished basement with average or better exposure
- Full bath in the basement
- Fireplace
- At least a 2-car garage
- 2 or more full bathrooms in the house
- A screened porch, with larger sizes being more valued
- A paved driveway

**Recommendations**

Buyers should know that they will need to pay more for larger homes, larger lots, newer and better-quality homes, all else being equal.

Sellers should focus on maintaining their house and property, and making improvements related to the key features listed above.
