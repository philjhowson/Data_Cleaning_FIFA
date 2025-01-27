import pandas as pd

data = pd.read_csv('data/raw_data/fifa21 raw data v2.csv')

"""
this is a large dataset, so I want to find just the columns with na values that need to be watched.
The only two columns are: ['Loan Date End', 'Hits']
"""

null_values = []
for column in data.columns:
    if data[column].isna().sum():
        null_values.append(column)

print(null_values)

"""
I know the column here contains na values, but I also checked and it contains no zeroes. So, here,
na values are converted to 0, because I'm not completely sure what it refers to, but possible hits
on the webpage, rather than strikes in the game (or other game related metric), so in this case
na would be a true 0.
"""

hits = data['Hits']

def into_numeric(string):
    if pd.isna(string):
        return 0
    if 'K' in str(string):
        string = float(string.replace('K', '')) * 1000
        return int(string)
    return int(string)

data['Hits'] = hits.apply(into_numeric)

"""
change height and weight column names to include the units. Remove the ' and " for feet/inch
measurements, and convert values into cm if they are in feet and inches, then convert the
column into int. This function works as is because we know no one is above 9'11" feet and below 1
foot in height.

For the weight column, create a function to convert into kg if the string contains 'lbs'. Convert to
float.
"""
data.rename(columns = {'Height' : 'Height (cm)', 'Weight' : 'Weight (kg)'}, inplace = True)

height = data['Height (cm)'].str.replace('\'', '').str.replace('"', '')

def into_cm(string):
    if 'cm' not in string:
        feet = float(string[0]) * 12
        total = float(string[1:]) + feet
        total = str(total * 2.54)
        return total + 'cm'
    return string

data['Height (cm)'] = height.apply(into_cm).str.replace('cm', '').astype(float).apply(lambda x : round(x, 0)).astype(int)

def into_kg(string):
    if 'lbs' in string:
        new_string = string.split('l')[0]
        new_value = str(int(new_string) * 0.454) + 'kg'
        return new_value
    return string

data['Weight (kg)'] = data['Weight (kg)'].apply(into_kg).str.replace('kg', '').astype(float)

"""
convert the joined column to datetime, and then create seperate columns for
day, month, year joined and drop the 'Joined' column.
"""
data['Joined'] = pd.to_datetime(data['Joined'], format='%b %d, %Y')

data['Year Joined'] = data['Joined'].dt.year
data['Month Joined'] = data['Joined'].dt.year
data['Day Joined'] = data['Joined'].dt.day

data.drop(columns = ['Joined'], inplace = True)

"""
normalize the contract dataframe so that contracts that start with Month + Day no longer have that, and
now start with year. Split the column and take the first item to get the starting year or 'free' for free
agents. Calculate the average start year and give free agents that as their start year. Create new columns
(0, 1) for if a player is on loan, and for 'Free Agent' status. Drop the original 'Contract' column.
"""

start = data['Contract'].str.replace(r'^[A-Za-z]+ \d{1,2}, ', '', regex = True).apply(lambda x : x.split(' ')[0])
mean_start = pd.to_numeric(start, errors = 'coerce').mean().astype(int)
data['Start Date'] = pd.to_numeric(start, errors = 'coerce').fillna(mean_start).astype(int)
data['On Loan'] = data['Contract'].str.contains('On Loan', na = False).astype(int)
data['Free Agent'] = data['Contract'].str.contains('Free', na = False).astype(int)

data.drop(columns = ['Contract'], inplace = True)

"""
replace the € symbol with nothing, create a simple function to determine if there is a K in the
string, indicating their value is less than €1M, then adds a . at the front of those values.
Remove 'M' and 'K' and convert to float and droop the original 'Value' column.
"""

value = data['Value'].str.replace('€', '')

def into_millions(string):
    if 'K' in string:
        return '.' + string
    return string

value = value.apply(into_millions)
data['Value (€M)'] = value.str.replace('M', '').str.replace('K', '').astype(float)
data.drop(columns = ['Value'], inplace = True)

"""
get ride € and K characters in the 'Wage' column and convert to float in order to get
the 'Wage (€K)' column. Drop the original column.
"""

data['Wage (€K)'] = data['Wage'].str.replace('€', '').str.replace('K', '').astype(float)
data.drop(columns = ['Wage'], inplace = True)

"""
Remove the € symbol from the 'Release Clause' and apply the function into_millions. Remove
'M' and 'K' from the columns, convert it to float, and make new column, 'Release Clause (€M)'.
Drop the original column.
"""

value = data['Release Clause'].str.replace('€', '')
value = value.apply(into_millions)
data['Release Clause (€M)'] = value.str.replace('M', '').str.replace('K', '').astype(float)
data.drop(columns = ['Release Clause'], inplace = True)

"""
remove the '★' from the 'W/F', 'SM', and 'IR' columns and the '\n'
from the 'Club' column.
"""

data['W/F'] = data['W/F'].str.replace(' ★', '').astype(int)
data['SM'] = data['SM'].str.replace('★', '').astype(int)
data['IR'] = data['IR'].str.replace(' ★', '').astype(int)
data['Club'] = data['Club'].str.replace('\n', '')

data.to_csv('data/processed_data/cleaned_data.csv', index = False)
