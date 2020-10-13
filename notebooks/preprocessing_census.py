import numpy as np
import pandas as pd


def open_datasets():
    label_file_path = "../input/us_census_full/labels.txt"
    with open(label_file_path) as f:
        names = [name.strip().replace(' ', '_') for name in f.readlines()]

    train_path = "../input/us_census_full/census_income_learn.csv"
    test_path = "../input/us_census_full/census_income_test.csv"
    train = pd.read_csv(train_path, names=names, sep=", ")
    test = pd.read_csv(test_path, names=names, sep=", ")

    train["income"].loc[(train["income"] == "50000+.")] = 1
    train["income"].loc[(train["income"] == "- 50000.")] = 0
    test["income"].loc[(test["income"] == "50000+.")] = 1
    test["income"].loc[(test["income"] == "- 50000.")] = 0
    
    train["income"] = train["income"].astype(int)
    
    return train, test


def get_missing_data(df):
    # replace 'Not in universe', 'Not identifiable' and '?' by null
    df.replace("Not in universe", np.nan, inplace=True)
    df.replace("Not identifiable", np.nan, inplace=True)
    df.replace("?", np.nan, inplace=True)

    total = df.isnull().sum()
    percent = (df.isnull().sum()/df.shape[0]*100)
    missing_values = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        types.append(dtype)
    missing_values["Types"] = types
    missing_values.sort_values('Total',ascending=False,inplace=True)
    
    return(missing_values)


def feature_transform(df, empty_cols):
    
    # Creating a categorical variable for age
    df["ageCat"] = 0
    df.loc[(df["age"] > 16) & (df["age"] <= 32), "ageCat"] = 1
    df.loc[(df["age"] > 32) & (df["age"] <= 48), "ageCat"] = 2
    df.loc[(df["age"] > 48) & (df["age"] <= 64), "ageCat"] = 3
    df.loc[(df["age"] > 64, "ageCat")] = 4
    
    # Creating a categorical variable for hispanic origin
    df["hispanicCat"] = 1
    df["hispanicCat"].loc[(df["hispanic_origin"] == "All other")] = 0
    df["hispanicCat"].loc[(df["hispanic_origin"].isna())] = 0
    
    # Creating a categorical variable to tell if the passenger is a Young/Mature/Senior male or a Young/Mature/Senior female
    df["sexCat"] = ""
    df["sexCat"].loc[(df["sex"] == "Male") & (df["age"] <= 21)] = "youngmale"
    df["sexCat"].loc[(df["sex"] == "Male") & ((df["age"] > 21) & (df["age"]) < 50)] = "maturemale"
    df["sexCat"].loc[(df["sex"] == "Male") & (df["age"] > 50)] = "seniormale"
    df["sexCat"].loc[(df["sex"] == "Female") & (df["age"] <= 21)] = "youngfemale"
    df["sexCat"].loc[(df["sex"] == "Female") & ((df["age"] > 21) & (df["age"]) < 50)] = "maturefemale"
    df["sexCat"].loc[(df["sex"] == "Female") & (df["age"] > 50)] = "seniorfemale"
    
    # Creating a categorical variable for unemployment
    df["unemployment"] = 0
    df["unemployment"].loc[(df["reason_for_unemployment"] == "Not in universe")] = 1
    
    # Creating "Household Frequency" Feature
    #There are too many detailded household categories
    df["household_Frequency"] = df.groupby("detailed_household_and_family_stat")["detailed_household_and_family_stat"].transform("count")
    
    # Droping empty column
    df.drop(empty_cols, axis=1, inplace=True)
    
    # Dropping unused columns from the feature set
    df.drop(["hispanic_origin",
             "detailed_household_and_family_stat", #too much categories
             "country_of_birth_father", #too much categories
             "country_of_birth_mother", #too much categories
             "country_of_birth_self" #too much categories
            ], axis=1, inplace=True)
    
    target = df["income"]
    df.drop(["income"], axis=1, inplace=True)
    
    # Splitting categorical and numerical column dataframes
    categorical_df = df.select_dtypes(include=['object'])
    numeric_df = df.select_dtypes(exclude=['object'])
    
    # And then, storing the names of categorical and numerical columns.
    categorical_columns = list(categorical_df.columns)
    numeric_columns = list(numeric_df.columns)
    
    print("Categorical columns:\n", categorical_columns)
    print("\nNumeric columns:\n", numeric_columns)

    return df, target, categorical_columns, numeric_columns


