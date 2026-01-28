# Week 2: Required Assignment 2.1

## 1) Understanding Raw Data

### Dataset Structure
There are 891 rows and 12 columns.

### Example Dataset
| PassengerId | Survived | Pclass | Name | Sex | Age | SibSp | Parch | Ticket | Fare | Cabin | Embarked |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 3 | Braund, Mr. Owen | male | 22 | 1 | 0 | A/5 21171 | 7.25 |  | S |
| 2 | 1 | 1 | Cumings, Mrs. John | female | 38 | 1 | 0 | PC 17599 | 71.2833 | C85 | C |
| 3 | 1 | 3 | Heikkinen, Miss. Laina | female | 26 | 0 | 0 | STON/O2. 3101282 | 7.925 |  | S |
| 4 | 1 | 1 | Futrelle, Mrs. Jacques | female | 35 | 1 | 0 | 113803 | 53.1 | C123 | S |
| 5 | 0 | 3 | Allen, Mr. William | male |  | 0 | 0 | 373450 | 8.05 |  | S |

### Column Names and Descriptions
| Column | Description |
| --- | --- |
| PassengerId | Unique ID for each passenger |
| Survived | 1 = survived, 0 = not survived |
| Pclass | Passenger class |
| Name | Full name |
| Sex | Gender |
| Age | Age in years |
| SibSp | Siblings or spouse aboard |
| Parch | Parents or children aboard |
| Ticket | Ticket number |
| Fare | Fare paid |
| Cabin | Cabin ID |
| Embarked | Port where passenger boarded |

### Common Issues in Raw Data
| Issue | Count / Example | Notes |
| --- | --- | --- |
| Missing values in Age | 177 | Imputation (fill missing) |
| Missing values in Cabin | 687 | Too many missing |
| Missing values in Embarked | 2 | Imputation with mode |
| Duplicate records | 0 | No duplicates |
| Different number sizes | Fare vs SibSp | Scaling needed |
| Very big values | Fare, Age | Outliers |
| Text columns | Name, Ticket | Hard for simple model |

## 2) Data Cleaning Techniques

### Handling Missing Values
I keep useful columns and remove very high missing ones. Age is kept and I fill with median (imputation). Embarked is kept and I fill with mode (imputation). Cabin is removed because too many missing.

### Example for Missing Values
Before: Age has blank like row 5.  
After: Fill Age with median value (imputation), for example Age = 28.  
Cabin has many blanks, so Cabin column is removed.

### Removing Non-Essential Columns
I remove PassengerId, Name, Ticket, and Cabin. PassengerId is only ID. Name and Ticket are text and hard to use. Cabin already removed. This keeps data simple.

### Example for Removing Columns
Before: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked.  
After: Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked.

## 3) Data Transformation

### Encoding Categorical Variables
I use encoding to change text columns to numbers. Sex is binary encoding (0/1). Embarked is one-hot encoding for C, Q, S.

### Example for Encoding
Sex: male -> 1, female -> 0.  
Embarked: C -> (1,0,0), Q -> (0,1,0), S -> (0,0,1).

### Feature Scaling
I scale numeric columns so one big value does not dominate. I use standardization for Age and Fare. I can also use normalization to make values between 0 and 1.

### Example for Scaling
Fare values like 7.25 and 71.2833 are very different. After standardization, they become values like -0.5 and 1.2 (example).

### Why Transformation Matters
Encoding and scaling make data ready for ML without changing meaning. It helps training and makes features comparable.

## 4) Reflection and Insights

### Challenges
Main challenges were learning new words and their meaning, like imputation (fill missing values), encoding (change text to numbers), scaling (make numbers similar size), and then using them correctly for this dataset.

### Importance of Each Step
Profiling gives structure and quality check. Handling missing values with imputation avoids errors. Choosing useful columns reduces noise. Encoding and scaling make data usable for ML.

## Python Code Snippet using pandas library
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Anand_V_Assignment-2-1.csv")

# drop non-essential columns
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# fill missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# encode categorical columns
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

# scale numeric columns
scaler = StandardScaler()
df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])
```
