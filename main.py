import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import euclidean_distances
from scipy.stats import pearsonr
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from scipy.stats import norm
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Load Dengue Fever cases dataset
dengue_cases = pd.read_csv('./Master_DB_v1.2.csv')\
                .loc[:, ['adm_0_name', 'Year', 'dengue_total']]\
                .groupby(['adm_0_name','Year']).sum().reset_index()\
                .rename(columns={'adm_0_name': 'Country'})\
                .assign(index=lambda x: x['Country'] + '_' + x['Year'].astype(str))\
                .set_index(['index'])\
                .loc[:, ['dengue_total']]

#.melt = convert data to dataframe and refactor year as column
#.loc = clear unececary columns
#.assign = convert country value to uppercase- preperation for merge
#.assign = creating another column contains the values of the country + year in order to preper an index
#.loc = leaving only the relevant feature column on the df

australia_temp = pd.read_csv('./features/Australlia/Annual_Surface_Temperature_Change.csv')\
                .melt(id_vars=['Country','ISO2','ISO3','Indicator','Unit','Source','CTS Code','CTS Name','CTS Full Descriptor'],
                      var_name='Year', value_name='australia_surface_temperature_change')\
                .loc[:, ['Country', 'Year', 'australia_surface_temperature_change']]\
                .assign(Country=lambda x: x['Country'].str.upper())\
                .assign(index=lambda x: x['Country'] + '_' + x['Year'].astype(str))\
                .set_index(['index'])\
                .loc[:, ['australia_surface_temperature_change']]
australia_co2 = pd.read_csv('./features/Australlia/CO2_Emissions_embodied_in_Domestic_Final_Demand_Production_and_Trade.csv')\
                .loc[lambda df: df['Indicator'] == 'CO2 Emissions Embodied in Production']\
                .melt(id_vars=['Country','ISO2','ISO3','Indicator','Unit','Source','CTS Code','CTS Name','CTS Full Descriptor','Scale'],
                      var_name='Year', value_name='australia_CO2_emissions')\
                .loc[:, ['Country', 'Year', 'australia_CO2_emissions']]\
                .assign(Country=lambda x: x['Country'].str.upper())\
                .assign(index=lambda x: x['Country'] + '_' + x['Year'].astype(str))\
                .set_index(['index'])\
                .loc[:, ['australia_CO2_emissions']]
australia_land = pd.read_csv('./features/Australlia/Land_Cover_Accounts.csv')\
                .loc[lambda df: df['Indicator'] == 'Tree-covered areas']\
                .melt(id_vars=['Country','ISO2','ISO3','Indicator','Unit','Source','CTS Code','CTS Name','CTS Full Descriptor','Climate Influence'],
                      var_name='Year', value_name='australia_Tree_covered_areas')\
                .loc[:, ['Country', 'Year', 'australia_Tree_covered_areas']]\
                .assign(Country=lambda x: x['Country'].str.upper())\
                .assign(index=lambda x: x['Country'] + '_' + x['Year'].astype(str))\
                .set_index(['index'])\
                .loc[:, ['australia_Tree_covered_areas']]
australia_solar_electricity = pd.read_csv('./features/Australlia/Renewable_Energy.csv')\
                .loc[lambda df: (df['Indicator'] == 'Electricity Generation') & (df['Technology'] == 'Solar energy')]\
                .melt(id_vars=['Country','ISO2','ISO3','Indicator','Technology','Energy Type','Unit','Source','CTS Name','CTS Code','CTS Full Descriptor'],
                      var_name='Year', value_name='australia_solar_electricity')\
                .loc[:, ['Country', 'Year', 'australia_solar_electricity']]\
                .assign(Country=lambda x: x['Country'].str.upper())\
                .assign(index=lambda x: x['Country'] + '_' + x['Year'].astype(str))\
                .set_index(['index'])\
                .loc[:, ['australia_solar_electricity']]

brazil_temp = pd.read_csv('./features/Brazil/Annual_Surface_Temperature_Change.csv')\
                .melt(id_vars=['Country','ISO2','ISO3','Indicator','Unit','Source','CTS Code','CTS Name','CTS Full Descriptor'],
                      var_name='Year', value_name='brazil_surface_temperature_change')\
                .loc[:, ['Country', 'Year', 'brazil_surface_temperature_change']]\
                .assign(Country=lambda x: x['Country'].str.upper())\
                .assign(index=lambda x: x['Country'] + '_' + x['Year'].astype(str))\
                .set_index(['index'])\
                .loc[:, ['brazil_surface_temperature_change']]
brazil_CO2 = pd.read_csv('./features/Brazil/CO2_Emissions_embodied_in_Domestic_Final_Demand_Production_and_Trade.csv')\
                .loc[lambda df: df['Indicator'] == 'CO2 Emissions Embodied in Production']\
                .melt(id_vars=['Country','ISO2','ISO3','Indicator','Unit','Source','CTS Code','CTS Name','CTS Full Descriptor','Scale'],
                      var_name='Year', value_name='brazil_CO2_emissions')\
                .loc[:, ['Country', 'Year', 'brazil_CO2_emissions']]\
                .assign(Country=lambda x: x['Country'].str.upper())\
                .assign(index=lambda x: x['Country'] + '_' + x['Year'].astype(str))\
                .set_index(['index'])\
                .loc[:, ['brazil_CO2_emissions']]
brazil_land = pd.read_csv('./features/Brazil/Land_Cover_Accounts.csv')\
                .loc[lambda df: df['Indicator'] == 'Tree-covered areas']\
                .melt(id_vars=['Country','ISO2','ISO3','Indicator','Unit','Source','CTS Code','CTS Name','CTS Full Descriptor','Climate Influence'],
                      var_name='Year', value_name='brazil_Tree_covered_areas')\
                .loc[:, ['Country', 'Year', 'brazil_Tree_covered_areas']]\
                .assign(Country=lambda x: x['Country'].str.upper())\
                .assign(index=lambda x: x['Country'] + '_' + x['Year'].astype(str))\
                .set_index(['index'])\
                .loc[:, ['brazil_Tree_covered_areas']]
brazil_solar_electricity = pd.read_csv('./features/Brazil/Renewable_Energy.csv')\
                .loc[lambda df: (df['Indicator'] == 'Electricity Generation') & (df['Technology'] == 'Solar energy')]\
                .melt(id_vars=['Country','ISO2','ISO3','Indicator','Technology','Energy Type','Unit','Source','CTS Name','CTS Code','CTS Full Descriptor'],
                      var_name='Year', value_name='brazil_solar_electricity')\
                .loc[:, ['Country', 'Year', 'brazil_solar_electricity']]\
                .assign(Country=lambda x: x['Country'].str.upper())\
                .assign(index=lambda x: x['Country'] + '_' + x['Year'].astype(str))\
                .set_index(['index'])\
                .loc[:, ['brazil_solar_electricity']]

cambodia_temp = pd.read_csv('./features/Cambodia/Annual_Surface_Temperature_Change.csv')\
                .melt(id_vars=['Country','ISO2','ISO3','Indicator','Unit','Source','CTS Code','CTS Name','CTS Full Descriptor'],
                      var_name='Year', value_name='cambodia_surface_temperature_change')\
                .loc[:, ['Country', 'Year', 'cambodia_surface_temperature_change']]\
                .assign(Country=lambda x: x['Country'].str.upper())\
                .assign(index=lambda x: x['Country'] + '_' + x['Year'].astype(str))\
                .set_index(['index'])\
                .loc[:, ['cambodia_surface_temperature_change']]
cambodia_CO2 = pd.read_csv('./features/Cambodia/CO2_Emissions_embodied_in_Domestic_Final_Demand_Production_and_Trade.csv') \
                .loc[lambda df: df['Indicator'] == 'CO2 Emissions Embodied in Production'] \
                .melt(id_vars=['Country','ISO2','ISO3','Indicator','Unit','Source','CTS Code','CTS Name','CTS Full Descriptor','Scale'],
                      var_name='Year', value_name='cambodia_CO2_emissions')\
                .loc[:, ['Country', 'Year', 'cambodia_CO2_emissions']]\
                .assign(Country=lambda x: x['Country'].str.upper())\
                .assign(index=lambda x: x['Country'] + '_' + x['Year'].astype(str))\
                .set_index(['index'])\
                .loc[:, ['cambodia_CO2_emissions']]
cambodia_land = pd.read_csv('./features/Cambodia/Land_Cover_Accounts.csv')\
                .loc[lambda df: df['Indicator'] == 'Tree-covered areas']\
                .melt(id_vars=['Country','ISO2','ISO3','Indicator','Unit','Source','CTS Code','CTS Name','CTS Full Descriptor','Climate Influence'],
                      var_name='Year', value_name='cambodia_Tree_covered_areas')\
                .loc[:, ['Country', 'Year', 'cambodia_Tree_covered_areas']]\
                .assign(Country=lambda x: x['Country'].str.upper())\
                .assign(index=lambda x: x['Country'] + '_' + x['Year'].astype(str))\
                .set_index(['index'])\
                .loc[:, ['cambodia_Tree_covered_areas']]
cambodia_solar_electricity = pd.read_csv('./features/Cambodia/Renewable_Energy.csv')\
                .loc[lambda df: (df['Indicator'] == 'Electricity Generation') & (df['Technology'] == 'Solar energy')]\
                .melt(id_vars=['Country','ISO2','ISO3','Indicator','Technology','Energy Type','Unit','Source','CTS Name','CTS Code','CTS Full Descriptor'],
                      var_name='Year', value_name='cambodia_solar_electricity')\
                .loc[:, ['Country', 'Year', 'cambodia_solar_electricity']]\
                .assign(Country=lambda x: x['Country'].str.upper())\
                .assign(index=lambda x: x['Country'] + '_' + x['Year'].astype(str))\
                .set_index(['index'])\
                .loc[:, ['cambodia_solar_electricity']]

#merged all the relevant df data into one df
merged_df = pd.merge(dengue_cases, australia_temp, left_index=True, right_index=True, how='outer')
merged_df = pd.merge(merged_df, brazil_temp, left_index=True, right_index=True, how='outer')
merged_df = pd.merge(merged_df, cambodia_temp, left_index=True, right_index=True, how='outer')
merged_df = pd.merge(merged_df, australia_co2, left_index=True, right_index=True, how='outer')
merged_df = pd.merge(merged_df, brazil_CO2, left_index=True, right_index=True, how='outer')
merged_df = pd.merge(merged_df, cambodia_CO2, left_index=True, right_index=True, how='outer')
merged_df = pd.merge(merged_df, australia_land, left_index=True, right_index=True, how='outer')
merged_df = pd.merge(merged_df, brazil_land, left_index=True, right_index=True, how='outer')
merged_df = pd.merge(merged_df, cambodia_land, left_index=True, right_index=True, how='outer')
merged_df = pd.merge(merged_df, australia_solar_electricity, left_index=True, right_index=True, how='outer')
merged_df = pd.merge(merged_df, brazil_solar_electricity, left_index=True, right_index=True, how='outer')
merged_df = pd.merge(merged_df, cambodia_solar_electricity, left_index=True, right_index=True, how='outer')

#combaining the columns of each country into one:

merged_df['surface_temperature_change'] = merged_df['australia_surface_temperature_change']\
                                          .fillna(merged_df['brazil_surface_temperature_change'])\
                                          .fillna(merged_df['cambodia_surface_temperature_change'])

merged_df['CO2_emissions'] = merged_df['australia_CO2_emissions']\
                                          .fillna(merged_df['brazil_CO2_emissions'])\
                                          .fillna(merged_df['cambodia_CO2_emissions'])

merged_df['Tree_covered_areas'] = merged_df['australia_Tree_covered_areas']\
                                          .fillna(merged_df['brazil_Tree_covered_areas'])\
                                          .fillna(merged_df['cambodia_Tree_covered_areas'])

merged_df['solar_electricity_use'] = merged_df['australia_solar_electricity']\
                                          .fillna(merged_df['brazil_solar_electricity'])\
                                          .fillna(merged_df['cambodia_solar_electricity'])

#final merge with the relevant columns only:
merged_df = merged_df.loc[:, ['dengue_total', 'surface_temperature_change', 'CO2_emissions', 'Tree_covered_areas', 'solar_electricity_use']]

print(merged_df)

#creating graphs with normalized values for each counrty seperatly to see the impact differance of each feature on dengue cases in each state:
normalized_df = merged_df.copy()
# Separate the data by state
states = normalized_df.index.str.split('_').str[0].unique()

#checking normality.
#If the p-value is greater than 0.05, it suggests that the feature is likely normally distributed.

for feature in ['surface_temperature_change', 'CO2_emissions', 'Tree_covered_areas', 'solar_electricity_use']:
    data = normalized_df[feature].dropna()
    stat, p = norm.fit(data)
    if p > 0.05:
        print(f"{feature} is likely normally distributed (p-value = {p:.4f})")
    else:
        print(f"{feature} is likely not normally distributed (p-value = {p:.4f})")

for state in states:
    state_data = normalized_df[normalized_df.index.str.startswith(state)]
    # Drop rows with missing values for dengue_total
    state_data = state_data.dropna(subset=['dengue_total'])
    # Normalize the feature values for the current state
    scaler = MinMaxScaler()
    features = ['dengue_total', 'surface_temperature_change', 'CO2_emissions', 'Tree_covered_areas', 'solar_electricity_use']
    state_data[features] = scaler.fit_transform(state_data[features])
    # Create a line chart with multiple lines
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(1, 1, 1)
    ax.plot(state_data.index, state_data['dengue_total'], label='Dengue Cases', color='red')
    for feature in ['surface_temperature_change', 'CO2_emissions', 'Tree_covered_areas', 'solar_electricity_use']:
        if feature in state_data.columns:
            ax.plot(state_data.index, state_data[feature], label=feature, linestyle='--')

    ax.set_xlabel('Year')
    ax.set_ylabel('Normalized Value')
    ax.set_title(f'{state} - Line Chart (Normalized)')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Create scatter plots for each climate change feature
    for feature in ['surface_temperature_change', 'CO2_emissions', 'Tree_covered_areas', 'solar_electricity_use']:
        if feature in state_data.columns:
            plt.figure(figsize=(8, 6))
            plt.scatter(state_data['dengue_total'], state_data[feature])
            plt.xlabel('Normalized Dengue Cases')
            plt.ylabel(f'Normalized {feature}')
            plt.title(f'{state} - {feature} Scatter Plot (Normalized)')
            non_na_feature = state_data.dropna(subset=['dengue_total', feature])
            correlation_coefficient, _ = pearsonr(non_na_feature['dengue_total'], non_na_feature[feature])
            plt.text(0.05, 0.95, f'Pearson r = {correlation_coefficient:.2f}', transform=plt.gca().transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            plt.tight_layout()
            plt.show()

    normalized_nona_df = state_data.dropna(
        subset=['dengue_total', 'surface_temperature_change', 'CO2_emissions', 'Tree_covered_areas',
                'solar_electricity_use'])
    normalized_nona_df = normalized_nona_df.dropna()
    # Create a distance matrix for dengue cases
    dengue_dist_matrix = euclidean_distances(normalized_nona_df['dengue_total'].values[:, None])
    print(f"State: {state}")
    print("=" * 30)
    # Perform Mantel test for each feature
    print("Mantel Test:")
    for feature in ['surface_temperature_change', 'CO2_emissions', 'Tree_covered_areas', 'solar_electricity_use']:
        if feature in normalized_nona_df.columns:
            # Create a distance matrix for the feature
            feature_dist_matrix = euclidean_distances(normalized_nona_df[feature].values[:, None])
            # Flatten the distance matrices and compute the Pearson correlation
            mantel_stat, p_value = pearsonr(dengue_dist_matrix.ravel(), feature_dist_matrix.ravel())

            print(f" Dengue Cases vs {feature}:")
            print(f"Mantel Statistic: {mantel_stat}")
            print(f"p-value: {p_value}")

        # Partial Correlation Analysis (PCA)
    print("\nPartial Correlation Analysis (PCA):")
    for feature in features:
        other_features = [f for f in features if f != feature]
        partial_corr, _ = pearsonr(normalized_nona_df['dengue_total'], normalized_nona_df[feature])
        print(f" Dengue Cases and {feature}: {partial_corr:.4f}")

    #Multiple Linear Regression:
    print("\nMultiple Linear Regression:")
    X = normalized_nona_df[['surface_temperature_change', 'CO2_emissions', 'Tree_covered_areas', 'solar_electricity_use']]
    y = normalized_nona_df['dengue_total']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(f"{model.summary()} \n")

    print("\n" + "=" * 30 + "\n")


#creating a prediction model
# split the data into training and testing sets:
train, test = train_test_split(merged_df, test_size=0.2, random_state=42)

# Define the feature and target variables for each country
australia_feature = 'solar_electricity_use'
brazil_feature = 'surface_temperature_change'
cambodia_feature = 'CO2_emissions'
target = 'dengue_total'

# Create separate models for each country
models = {}
accuracies = {}

for country, feature in zip(['AUSTRALIA', 'BRAZIL', 'CAMBODIA'], [australia_feature, brazil_feature, cambodia_feature]):
    # Filter the data for the current country
    country_train = train[train.index.str.startswith(country)]
    country_test = test[test.index.str.startswith(country)]

    # Impute missing values with the median
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(country_train[[feature]])
    X_test = imputer.transform(country_test[[feature]])

    # Impute missing values in the target variable
    y_train = country_train[target].fillna(country_train[target].median())
    y_test = country_test[target].fillna(country_test[target].median())

    # Create the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Store the model
    models[country] = model

    # Calculate the accuracy
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracies[country] = {'MSE': mse, 'R^2': r2}

# Ask the user for input
country = input("Enter the country (AUSTRALIA, BRAZIL, or CAMBODIA): ").upper()
year = int(input("Enter the future year: "))

# Select the appropriate feature and model
if country == 'AUSTRALIA':
    feature = australia_feature
    model = models['AUSTRALIA']
elif country == 'BRAZIL':
    feature = brazil_feature
    model = models['BRAZIL']
else:
    feature = cambodia_feature
    model = models['CAMBODIA']

# Forecast the feature value for the future year
country_data = merged_df[merged_df.index.str.startswith(country)]
country_data = country_data.dropna(subset=[feature])
X = country_data.index.str.extract(r'_(\d+)').astype(int)
y = country_data[feature]
linear_model = LinearRegression()
linear_model.fit(X, y)
feature_value = linear_model.predict([[year]])

# Make the prediction
prediction = model.predict([[feature_value[0]]])
print(f"Predicted number of dengue cases for {country} in {year}: {prediction[0]:.2f}")

# Print the accuracy metrics
print("\nAccuracy Metrics:")
for country, metrics in accuracies.items():
    print(f"{country}:")
    print(f"  Mean Squared Error: {metrics['MSE']:.2f}")
    print(f"  R^2 Score: {metrics['R^2']:.2f}")

train, test = train_test_split(merged_df, test_size=0.2, random_state=42)

# Create separate models for each country
models = {}
accuracies = {}

for country, feature in zip(['AUSTRALIA', 'BRAZIL', 'CAMBODIA'], [australia_feature, brazil_feature, cambodia_feature]):
    # Filter the data for the current country
    country_train = train[train.index.str.startswith(country)]
    country_test = test[test.index.str.startswith(country)]

    # Impute missing values with the median
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(country_train[[feature]])
    X_test = imputer.transform(country_test[[feature]])

    # Impute missing values in the target variable
    y_train = country_train['dengue_total'].fillna(country_train['dengue_total'].median())
    y_test = country_test['dengue_total'].fillna(country_test['dengue_total'].median())

    # Create the XGBoost model
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Store the model
    models[country] = model

    # Calculate the accuracy
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracies[country] = {'MSE': mse, 'R^2': r2}

# Ask the user for input
country = input("Enter the country (AUSTRALIA, BRAZIL, or CAMBODIA): ").upper()
year = int(input("Enter the future year: "))

# Make the prediction
model = models[country]
prediction = model.predict([[feature_value[0]]])
print(f"Predicted number of dengue cases for {country} in {year}: {prediction[0]:.2f}")

# Print the accuracy metrics
print("\nAccuracy Metrics:")
for country, metrics in accuracies.items():
    print(f"{country}:")
    print(f"  Mean Squared Error: {metrics['MSE']:.2f}")
    print(f"  R^2 Score: {metrics['R^2']:.2f}")