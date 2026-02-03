# AIRCRAFT ACCIDENT ANALYSIS AND SAFETY INSIGHTS


## BUSINESS UNDERSTANDING
From a business / aviation safety perspective, stakeholders want to know:

- Are accidents increasing or decreasing over time?

- How severe are most accidents?

- How often do accidents involve fatalities?

- Which operators appear most frequently in accident reports?

These insights can help:

- Airlines improve safety procedures

- Regulators focus inspections

- Insurance companies assess risk

## OBJECTIVES
- Understand accident trends over time

- Identify damage severity patterns

- Examine fatal vs non-fatal accidents

- Identify operators with higher accident counts

```python
# import all the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
# loading the dataset
df = pd.read_csv('/content/flight.csv')
df
```

## DATA UNDERSTANDING

*The dataset contains 2,500 flight accident records with the following columns:*

*'Unnamed', 'acc.date', 'type', 'reg', 'operator', 'fat', 'location', 'dmg'.*

*Most columns are categorical.*

```python
# Checking for missing values
df.isna().sum()
```

*we have 92 missing values in 'reg column', 14 in 'operator column' and 12 in 'fat column'. The rest have no missing values.*

## DATA PREPERATION AND CLEANING

*This step ensures the data is usable and reliable.*

```python
# Remove unnecessary index column 'Unnamed: 0'
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])
df
```

```python
# Convert accident date to datetime
df['acc.date'] = pd.to_datetime(df['acc.date'], errors='coerce')
df
```

 <img width="1674" height="542" alt="Screenshot 2026-02-02 210038" src="https://github.com/user-attachments/assets/75169d14-967b-42ac-823e-1d216000c19b" />

- Handling missing values

*Since this is a sensitive data, there are various ways to handle the missing data/values. So i've decided to handle each of them by the column they are in.*

1. Accident Date (acc.date), drop if it has missing values. without date record can't be used for trend or time based analysis.

```python
# dropping missing values in acc.date column
df = df.dropna(subset=['acc.date'])
```

2. Reg and Operator columns, to be filled with 'unknown'. dropping them would lose valuable incidents.

```python
# Replacing missing values with 'unknown'
categorical_cols = ['reg', 'operator']

for col in categorical_cols:
    df[col] = df[col].fillna('Unknown')
```

3. Fatalities(fat) column best approach is median imputation since median is robust and realistic, and may contain outliers.

```python
#using the .loc method and median to fill the missing values in the fatalities column
df.loc[:, 'fat'] = pd.to_numeric(df['fat'], errors='coerce')
df.loc[:, 'fat'] = df['fat'].fillna(df['fat'].median())
```

<img width="838" height="265" alt="Screenshot 2026-02-02 210550" src="https://github.com/user-attachments/assets/66c0572c-9898-45bd-ba47-92958d30bb7e" />

```python
# Ensure there are no missing values left
df.isna().sum()
```

<img width="515" height="224" alt="Screenshot 2026-02-02 210826" src="https://github.com/user-attachments/assets/fff1e739-b4be-4409-bf7d-0152ed16e2c3" />

- Handling of Duplicates

```python
# finding the number of duplicates
df.duplicated().sum()
```

*There are 1247 duplicated values in this data set, which may be because this data may be extracted from many sources and combined to one.*

*Here we shall keep the most complete record (partial duplicates) and drop the rest*

```python
# the most complete duplicates will be left in the dataset
df['missing_count'] = df.isna().sum(axis=1)

df = df.sort_values('missing_count')
df = df.drop_duplicates(
    subset=['acc.date', 'type', 'location'],
    keep='first'
)

df = df.drop(columns='missing_count')
```

```python
# confirming that we have no duplicated values in the dataset
df.duplicated().sum()
```

We have zero duplicates left

# DATA ANALYSIS- EXPLORATORY DATA ANALYSIS(EDA)

- Finding out the number of Accidents per year and then visualizing it using a line chart.

<img width="573" height="455" alt="image" src="https://github.com/user-attachments/assets/ac81e03c-71b0-47c4-9a41-b6730895baec" />

*The line chart shows that the number of flight accidents peaked around 2018–2019 and declined afterward. This trend may be associated with reduced global air traffic during the COVID-19 pandemic.*

- EDA: Damage Severity Distribution. Here we look at accidents that resulted in substantial damage,write-offs, Minor and no-damage

<img width="695" height="470" alt="image" src="https://github.com/user-attachments/assets/da64776f-8e41-4103-8c6d-2deef55ebbb2" />

*The damage severity distribution shows that most recorded accidents resulted in substantial damage, followed by write-offs. Minor and no-damage cases are relatively rare, indicating that reported accidents often involve significant aircraft damage.*

- EDA: Fatal vs Non-Fatal Accidents using a pie chart visualization

<img width="484" height="504" alt="image" src="https://github.com/user-attachments/assets/92901f3f-efc0-4c13-b8c9-d654110b6db9" />

*The visualization shows that the majority of flight accidents are non-fatal. Fatal accidents represent a smaller proportion of total incidents, indicating that while accidents occur, loss of life is relatively infrequent.*

- EDA: Top 10 Operators by Accident Count

<img width="952" height="470" alt="image" src="https://github.com/user-attachments/assets/969c01a8-52b1-4e4f-98e3-dc0be814c846" />

*The visualization shows that private operators account for the highest number of recorded accidents. This may be due to a large number of small private flights rather than poorer safety performance. Therefore, accident counts should be normalized by flight volume for fair comparison.*

- EDA Most Frequent Accident Locations

<img width="932" height="455" alt="image" src="https://github.com/user-attachments/assets/81724202-200f-4bd7-9da4-38aed723fca7" />

- EDA Distribution of Fatalities per Accident using a Histogram

<img width="580" height="455" alt="image" src="https://github.com/user-attachments/assets/d2b4e0f2-4ae6-4438-9f29-9101b9b56662" />

*Here we can see The distribution is heavily right-skewed Most accidents occur at 0 fatalities*

*Most flight accidents do not result in loss of life, but when fatalities occur, they can be severe.*

- FINDING OUT THE CORRELATION BETWEEN DAMAGE TYPE AND FATALITIES

<img width="515" height="435" alt="image" src="https://github.com/user-attachments/assets/408fe3f9-737f-4adf-a4a9-312f110c66fb" />

*Higher damage severity correlates with fatalities. The more the damage to an aircraft the more its likely to have fatalities*

- Finding out the Number of Accidents by Month

<img width="850" height="470" alt="image" src="https://github.com/user-attachments/assets/e6129c7e-9310-4990-85f5-0a1bd46d8fda" />

*The line chart shows the distribution of flight accidents across months. Variations in accident counts suggest seasonality, possibly influenced by weather conditions, peak travel periods, and operational factors.*

For further Analysis consult my Tableau dashboard here: https://public.tableau.com/views/projectphase1_17700368621660/AccidentTrendsOverTimeandSeasonality?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link

# CONCLUSION

1. Accident Frequency Has Declined in Recent Years. The yearly trend shows a reduction in accidents, especially after 2019.
2. Most Accidents Are Non-Fatal. The fatal vs non-fatal analysis shows that the majority of accidents do not result in fatalities.
3. Substantial and Write-Off Damage Is Common. Damage severity analysis shows that most reported accidents involve significant aircraft damage.
4. Accident Frequency Varies by Month. Environmental and operational factors influence accident occurrence.
5. Private Operators Appear Frequently in Accident Records. This may be due to Larger number of small private flights.

# RECOMMENDATIONS

1. Seasonal Risk Mitigation. Increase inspections, crew training, and operational caution during high-risk months.
2. Improve Safety Oversight for Private Operators. Strengthen regulations and audits for private and charter operators.
3. Enhance Data Collection to include Weather conditions, Aircraft age. Richer data enables better predictive modeling and policy decisions.

*The analysis indicates that while flight accidents still occur, the majority are non-fatal and accident frequency has declined over time. However, a small number of catastrophic events account for a disproportionate share of fatalities. Seasonal patterns and operator characteristics further influence accident occurrence. These findings suggest that aviation safety efforts should focus not only on reducing accident frequency but also on mitigating the severity of rare high-impact events.*










    
  

    
  
