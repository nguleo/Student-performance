
import pandas as pd
import numpy as np

def identify_inconsistencies(student_math_grade):
    '''This function will check if a column is categorical and print the unique values in the column.'''
    for column in student_math_grade.columns:
        print(f'{column}: {student_math_grade[column].nunique()} unique values')
        print(student_math_grade[column].unique())
    return student_math_grade

def handle_inconsistencies(student_math_grade):
    '''This function will handle inconsistencies in the data.'''
    categorical_columns = student_math_grade.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        student_math_grade[column] = student_math_grade[column].str.upper().str.strip()
    return student_math_grade

def check_missing_values(student_math_grade):
    '''This function will check for missing values in the data.'''
    print('---Checking for missing values in the data---')
    print('Number of missing values in each column:')
    print(student_math_grade.isnull().sum())

def impute_missing_values(student_math_grade):
    '''This function will impute missing values in numerical columns with -1 and in categorical columns with 'Unknown'.'''
    print('---Imputing missing values in the data---')
    for column in student_math_grade.columns:
        if student_math_grade[column].dtype == 'object':
            student_math_grade[column] = student_math_grade[column].fillna('Unknown')
        else:
            student_math_grade[column] = student_math_grade[column].fillna(-1)
    return student_math_grade

def check_duplicate_columns(student_math_grade):
    '''This function will check if there are any duplicate columns in the data.'''
    duplicate_columns = []
    for x in range(student_math_grade.shape[1]):
        col = student_math_grade.iloc[:, x]
        for y in range(x + 1, student_math_grade.shape[1]):
            other_col = student_math_grade.iloc[:, y]
            if col.equals(other_col):
                duplicate_columns.append(student_math_grade.columns.values[y])
    return duplicate_columns

def check_duplicate_rows(student_math_grade):
    '''This function will check if there are any duplicate rows in the data.'''
    print('---Checking for duplicate rows in the data---')
    print("Number of duplicate rows:", student_math_grade.duplicated().sum())

def detect_outliers(student_math_grade):
    '''This function will detect outliers using the IQR method.'''
    outliers_info = {}
    for column in student_math_grade.select_dtypes(include=['number']).columns:
        Q1 = student_math_grade[column].quantile(0.25)
        Q3 = student_math_grade[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = student_math_grade[(student_math_grade[column] < lower_bound) | (student_math_grade[column] > upper_bound)]
        if not outliers.empty:
            print(f'Outliers detected in {column}')
            print(f'Lower bound: {lower_bound}, Upper bound: {upper_bound}')
            outliers_info[column] = (lower_bound, upper_bound)
    return outliers_info

def handle_outliers(student_math_grade, outlier_info):
    '''This function will handle outliers by capping them to the lower and upper bounds.'''
    print('---Handling outliers by replacing outliers with -1---')
    for column, (lower_bound, upper_bound) in outlier_info.items():
        student_math_grade[column] = np.where(student_math_grade[column] < lower_bound, -1, student_math_grade[column])
        student_math_grade[column] = np.where(student_math_grade[column] > upper_bound, -1, student_math_grade[column])
    return student_math_grade

def identify_highly_correlated_features(student_math_grade):
    '''This function will identify highly correlated features in the data.'''
    print('---Identifying highly correlated features in the data---')
    num_cols = student_math_grade.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = student_math_grade[num_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
    print("Highly correlated columns:", to_drop)
    return to_drop

def data_cleaning(student_math_grade):
    '''This function will clean the data by removing any missing values.'''
    # 1. Check for inconsistencies
    identify_inconsistencies(student_math_grade)

    # 2. Handle inconsistencies
    cleaned_data = handle_inconsistencies(student_math_grade)

    # 3. Check and handle missing values
    check_missing_values(cleaned_data)
    cleaned_data = impute_missing_values(cleaned_data)

    # 4. Check and handle duplicate columns
    #duplicate_cols = check_duplicate_columns(cleaned_data)
    #cleaned_data = cleaned_data.drop(columns=duplicate_cols)

    # 5. Check and handle duplicate rows
    #check_duplicate_rows(cleaned_data)
    #cleaned_data = cleaned_data.drop_duplicates()

    # 6. Check and handle outliers
    #outlier_cols = detect_outliers(cleaned_data)
    #cleaned_data = handle_outliers(cleaned_data, outlier_cols)

    # 7. Check for highly correlated features
    #corr_cols = identify_highly_correlated_features(cleaned_data)
    #cleaned_data = cleaned_data.drop(columns=corr_cols)

    return cleaned_data

def main():
    data = pd.read_csv('StudentMathGradeDataset.csv')
    cleaned_data = data_cleaning(data)
    cleaned_data.to_csv('cleaned_StudentMathGradeDataset.csv', index=False)

# Call the main function
main()