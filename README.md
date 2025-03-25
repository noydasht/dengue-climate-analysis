**The research question:**

\*Measure the Impact of climate change features on dengue fever cases in 3 different areas: states from Asia (Cambodia), south America (Brazil) and Australia.   
\*Predict the dengue cases in the future in each state according to the climate change features tendency.

**The data:**

1.  Master_DB_v1.2.csv  
    taken from OpenDengue project - <https://opendengue.org/>  
    A global database of publicly available dengue case data.   
    This data includes info regarding the state and number of Confirmed patients in each month.

| **Column name**               | **Description**                                                                                                                                                                                                                                                                                                                        |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| adm_0_name                    | National name                                                                                                                                                                                                                                                                                                                          |
| adm_1_name                    | Sub national name                                                                                                                                                                                                                                                                                                                      |
| adm_2_name                    | Sub national name                                                                                                                                                                                                                                                                                                                      |
| full_name                     | Full country name                                                                                                                                                                                                                                                                                                                      |
| ISO_A0                        |  [Natural Earth](https://www.naturalearthdata.com/) shapefiles country name adaptation                                                                                                                                                                                                                                                 |
| FAO_GAUL_code                 | Global Administrative Unit Layers (GAUL, see [UN FAO website](https://data.apps.fao.org/catalog/dataset/gaul-codes)) codes country name adaptation                                                                                                                                                                                     |
| RNE_iso_code                  | [Natural Earth](https://www.naturalearthdata.com/) shapefiles sub national name adaptation                                                                                                                                                                                                                                             |
| IBGE_code                     | Adress format                                                                                                                                                                                                                                                                                                                          |
| calendar_start_date           | Time periods are recorded using a start date (calendar_start_date) and end date (calendar_end_date) with the format YYYY-mm-dd.                                                                                                                                                                                                        |
| calendar_end_date             | Time periods are recorded using a start date (calendar_start_date) and end date (calendar_end_date) with the format YYYY-mm-dd.                                                                                                                                                                                                        |
| year                          | Year of counting                                                                                                                                                                                                                                                                                                                       |
| dengue_total                  | Number of cases for each time frame (start_day-end_day)                                                                                                                                                                                                                                                                                |
| case_definition_standardised  | methods of confirmation (suspected, probable, clinically confirmed, laboratory confirmed)                                                                                                                                                                                                                                              |
| S_res                         | Global Administrative Unit Layers codes                                                                                                                                                                                                                                                                                                |
| T_res                         | Calendar type (year/month/day)                                                                                                                                                                                                                                                                                                         |
| UUID                          | The specific case definition used for a particular data point will depend on the original data source and can be identified by looking up the record case_definition_original in the sourcedata_V1.2.csv file by the UUID which can be found in the [OpenDengue Github repo](https://github.com/OpenDengue/master-repo/data/raw_data). |

For each country, used the following csv files, Taken from IMF (12 files overall)-<https://climatedata.imf.org/> Climate Change Indicators Dashboard

2.  Annual_Surface_Temperature_Change.csv:

    Represent Temperature change with respect to a baseline climatology, corresponding to the period 1951-1980 in Celsius degrees per year.

3.  CO2_Emissions_embodied_in_Domestic_Final_Demand_Production_and_Trade.csv:

    Represent CO2 Emissions Embodied in Production in Millions of metric tons units

4.  Land_Cover_Accounts.csv:   
    Represent tree-covered areas in each country in 1000 HA units over the years.

5.  Renewable_Energy.csv:

    Represent the use of renewable energy (Electricity Generation by solar energy), counting in Gigawatt-hours (GWh) units.

    **introduction about the problem:**

    Dengue fever is a programmed tropical disease transmitted by Aedes genus mosquitoes that can be found in tropical regions around the world. The disease is particularly common in tropical and subtropical regions, as well as among travelers. Dengue fever is currently widespread in over 110 countries over the world.   
    Every year around 50 to 528 million people become infected with dengue fever and about 10,000-20,000 die from it.

    The symptoms of dengue fever usually continue 3 to 14 days after infection and they may include high fever, headache, vomiting, muscle and joint pain, and skin rash. Recovery usually takes two to seven days.

    According to data from the World Health Organization, dengue fever infections have increased in recent years. currently, more than half of the world's population are at risk to be infected. As a result, World Health Organization defined this disease as one of the ten greatest threats to human health.  
    There are four strains of dengue virus. Dengue hemorrhagic fever (an escalation of the symptoms of the disease) usually occurs in people who have recovered from infection with one strain of the virus but are later reinfected with another strain of the virus. The reason is not totally sure. Common theory is that instead of fighting the second strain, the immune system produces a lot of antibodies against the first strain.  
    Until recently, there was no vaccine for dengue fever. In 2016, the first vaccine for was developed, but It’s still not used worldwide.   
    I have been infected by dengue on my trip to Cambodia back in 2019.  
    The facts that there is a risk for escalated symptoms by second bite, and there is no vaccine- are the reasons why it’s important to me personally and generally to know the chance to get infected by dengue. It’s important to investigate the spreading of the disease on different locations over time, knowing the weather (infected by climate change) is crucial factor for the infected mosquito to exist.

    \-Methods:

    \-**using pandas** I have build a table contains all the relevant info regarding the dengue cases and each feature.

    \-**using MinMaxScaler()** function, I have normalized each feature values to dengue cases in each state (since each data contains different scale units).

    \-**using** **mantel test** I have tried to understand each feature has the most impact on dengue cases on all countries. It seems like this test is not suitable for this case since it’s for non-linear data and according to the graphs, it seems like there is a linear correlation for most of the cases.   
    \-**using dropna()** for missing values. Type of missingness is missing completely at random. (not each year contains info in all of the features measurements).

    \-**using PCA** for understand the relationships between dengue fever cases and environmental factors.

    \-**using Multiple Linear Regression** to examine the relationship between the different features on dengue cases.

    \-**using train_test_split** for splitting the data and test the model

    \-**using RandomForestRegressor** model for machine learning algorithm

    \- **linear_model** to predict the feature value in the future.

    \-fillna with median value for using the algorithm.   
    \- **XGBoost model**

    \-Results:

    **Graphs:**

    ![A graph of different colored lines Description automatically generated](72911850d019cd68671d17fd262c5eb2.png)

    ![A graph of a graph with blue dots Description automatically generated](42ba61a5d2864cbccbe82142dcf44175.png)![A graph of a virus Description automatically generated with medium confidence](e8f2185eef9e4f650e56d847d56a16f2.png)

    ![A graph with blue dots Description automatically generated](3c015b71bc36d8c86ff08c761d307419.png)![A graph with blue dots Description automatically generated](17681df1a5d47a827e87a85493e1fced.png)

    ![A graph of lines and numbers Description automatically generated](7bfe8905173bd572fb624f9c4ff70747.png)

    ![A graph with blue dots Description automatically generated](dfc06329cc4a973d42baa9e0f75fcab7.png)![A graph with blue dots Description automatically generated](fd91498507c102985152c5050f4a1b4f.png)

    ![A graph of a virus Description automatically generated with medium confidence](8eec2e6af0e6a4c26529553f2380d263.png)![A graph with blue dots Description automatically generated](6721b42e9cf7abf0cb1bf661b87675ff.png)

![A graph of different colored lines Description automatically generated](faabb8b5ff4cf76914cf339883688abd.png)

![A screen shot of a graph Description automatically generated](d38d5c47e92da604ca4f8201b157dea6.png)![A screen shot of a graph Description automatically generated](b089d99f725076cc45e80ddfa9200fcd.png)

![A graph with blue dots Description automatically generated](cbd6d1d177ecc8408b614f78f5edbda1.png)![A graph with blue dots Description automatically generated](6d65da08cd75609b667b483c9305c542.png)

Mantel test + PCA + linear regression:

**State: AUSTRALIA**

**According to the graphs, it looks like the most correlated feature with dengue fever cases Is CO2 emission. It’s not kept in line with mantel\\PCA results.**

==============================

Mantel Test:

Dengue Cases vs surface_temperature_change:

Mantel Statistic: 0.22774290577539535

p-value: 1.2443814711185143e-05

Dengue Cases vs CO2_emissions:

Mantel Statistic: 0.2056877491049044

p-value: 8.261428796907696e-05

Dengue Cases vs Tree_covered_areas:

Mantel Statistic: 0.05017965618460105

p-value: 0.3417536673418326

Dengue Cases vs solar_electricity_use:

Mantel Statistic: 0.33940182147291975

p-value: 3.498177812577795e-11

Partial Correlation Analysis (PCA):

Dengue Cases and dengue_total: 1.0000

Dengue Cases and surface_temperature_change: 0.4800

Dengue Cases and CO2_emissions: 0.5481

Dengue Cases and Tree_covered_areas: -0.3108

Dengue Cases and solar_electricity_use: 0.5691

Multiple Linear Regression:

OLS Regression Results

==============================================================================

Dep. Variable: dengue_total R-squared: 0.527

Model: OLS Adj. R-squared: 0.392

Method: Least Squares F-statistic: 3.899

Date: Thu, 09 May 2024 Prob (F-statistic): 0.0249

Time: 14:06:44 Log-Likelihood: 3.6886

No. Observations: 19 AIC: 2.623

Df Residuals: 14 BIC: 7.345

Df Model: 4

Covariance Type: nonrobust

==============================================================================================

coef std err t P\>\|t\| [0.025 0.975]

\----------------------------------------------------------------------------------------------

const -0.1769 0.290 -0.609 0.552 -0.799 0.446

surface_temperature_change 0.1528 0.230 0.663 0.518 -0.341 0.647

CO2_emissions 0.6717 0.380 1.768 0.099 -0.143 1.487

Tree_covered_areas -0.5377 0.375 -1.433 0.174 -1.342 0.267

solar_electricity_use 0.7875 0.603 1.307 0.212 -0.505 2.080

==============================================================================

Omnibus: 11.243 Durbin-Watson: 1.142

Prob(Omnibus): 0.004 Jarque-Bera (JB): 2.097

Skew: 0.067 Prob(JB): 0.351

Kurtosis: 1.378 Cond. No. 17.0

==============================

**State: BRAZIL**

==============================

Mantel Test:

Dengue Cases vs surface_temperature_change:

Mantel Statistic: 0.4214569735510175

p-value: 0.0025621719825444263

Dengue Cases vs CO2_emissions:

Mantel Statistic: 0.10899521623358271

p-value: 0.4559721864498283

Dengue Cases vs Tree_covered_areas:

Mantel Statistic: 0.15880471280424868

p-value: 0.275770403136535

Dengue Cases vs solar_electricity_use:

Mantel Statistic: 0.058881906894948814

p-value: 0.6877708763306848

Partial Correlation Analysis (PCA):

Dengue Cases and dengue_total: 1.0000

Dengue Cases and surface_temperature_change: 0.4738

Dengue Cases and CO2_emissions: 0.0507

Dengue Cases and Tree_covered_areas: -0.2281

Dengue Cases and solar_electricity_use: -0.4479

Multiple Linear Regression:

OLS Regression Results

==============================================================================

Dep. Variable: dengue_total R-squared: 0.503

Model: OLS Adj. R-squared: -0.490

Method: Least Squares F-statistic: 0.5064

Date: Thu, 09 May 2024 Prob (F-statistic): 0.747

Time: 14:06:46 Log-Likelihood: 1.1436

No. Observations: 7 AIC: 7.713

Df Residuals: 2 BIC: 7.442

Df Model: 4

Covariance Type: nonrobust

==============================================================================================

coef std err t P\>\|t\| [0.025 0.975]

\----------------------------------------------------------------------------------------------

const 1.0786 2.165 0.498 0.668 -8.237 10.395

surface_temperature_change 1.2421 1.284 0.967 0.435 -4.283 6.767

CO2_emissions -1.5683 2.417 -0.649 0.583 -11.970 8.833

Tree_covered_areas -17.9922 31.543 -0.570 0.626 -153.710 117.725

solar_electricity_use -0.6060 3.873 -0.156 0.890 -17.269 16.057

==============================================================================

Omnibus: nan Durbin-Watson: 2.901

Prob(Omnibus): nan Jarque-Bera (JB): 0.485

Skew: 0.227 Prob(JB): 0.785

Kurtosis: 1.793 Cond. No. 334.

==============================================================================

Notes:

[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

==============================

**State: CAMBODIA**

==============================

Mantel Test:

Dengue Cases vs surface_temperature_change:

Mantel Statistic: 0.04106898533386966

p-value: 0.4366086519833043

Dengue Cases vs CO2_emissions:

Mantel Statistic: 0.10776026450507784

p-value: 0.04072623352392493

Dengue Cases vs Tree_covered_areas:

Mantel Statistic: 0.1473712548421914

p-value: 0.005021088669522176

Dengue Cases vs solar_electricity_use:

Mantel Statistic: 0.10812250195942587

p-value: 0.04005069907771187

Partial Correlation Analysis (PCA):

Dengue Cases and dengue_total: 1.0000

Dengue Cases and surface_temperature_change: 0.2205

Dengue Cases and CO2_emissions: 0.3092

Dengue Cases and Tree_covered_areas: -0.4949

Dengue Cases and solar_electricity_use: 0.2487

Multiple Linear Regression:

OLS Regression Results

==============================================================================

Dep. Variable: dengue_total R-squared: 0.285

Model: OLS Adj. R-squared: 0.080

Method: Least Squares F-statistic: 1.393

Date: Thu, 09 May 2024 Prob (F-statistic): 0.287

Time: 14:06:47 Log-Likelihood: 22.261

No. Observations: 19 AIC: -34.52

Df Residuals: 14 BIC: -29.80

Df Model: 4

Covariance Type: nonrobust

==============================================================================================

coef std err t P\>\|t\| [0.025 0.975]

\----------------------------------------------------------------------------------------------

const 0.3091 0.153 2.021 0.063 -0.019 0.637

surface_temperature_change 0.0075 0.115 0.065 0.949 -0.240 0.255

CO2_emissions -0.4250 0.556 -0.765 0.457 -1.617 0.767

Tree_covered_areas -0.5612 0.330 -1.698 0.112 -1.270 0.148

solar_electricity_use 4.4857 6.674 0.672 0.512 -9.829 18.800

==============================================================================

Omnibus: 0.759 Durbin-Watson: 2.112

Prob(Omnibus): 0.684 Jarque-Bera (JB): 0.737

Skew: 0.393 Prob(JB): 0.692

Kurtosis: 2.440 Cond. No. 397.

***

Accuracy Metrics:

AUSTRALIA:

Mean Squared Error: 156432.56

R\^2 Score: 0.66

BRAZIL:

Mean Squared Error: 274837857745.20

R\^2 Score: -1.50

CAMBODIA:

Mean Squared Error: 1318157042.48

R\^2 Score: -0.07

![A screenshot of a computer error Description automatically generated](6e406e714666ba47227c31b76e9306ba.png)

– Discussion/conclusions:

**Australia:**

According to the Partial Correlation Analysis (PCA), the feature with the highest partial correlation with dengue cases is solar_electricity_use (0.5691).

The Mantel test also suggests a strong correlation between dengue cases and solar_electricity_use, with a high Mantel statistic (0.3394) and a very low p-value (3.498e-11).

In the Multiple Linear Regression, while Tree_covered_areas has the highest coefficient (2.6890), it is not statistically significant (p-value = 0.174).

Overall, the results indicate that **solar_electricity_use** is the most significant feature for dengue cases in Australia.

**Brazil:**

The Mantel test shows the strongest correlation between dengue cases and surface_temperature_change (Mantel statistic = 0.4215, p-value = 0.0026).

In the PCA, surface_temperature_change also has the highest partial correlation coefficient (0.4738).

However, the Multiple Linear Regression results are not statistically significant for any of the features.

Based on the Mantel test and PCA results, **surface_temperature_change** appears to be the most significant feature for dengue cases in Brazil.

**Cambodia:**

The Mantel test suggests significant correlations between dengue cases and CO2_emissions (p-value = 0.0407), Tree_covered_areas (p-value = 0.0050), and solar_electricity_use (p-value = 0.0401).

The PCA shows the highest partial correlation coefficient for Tree_covered_areas (-0.4949).

In the Multiple Linear Regression, none of the features are statistically significant, likely due to the small sample size.

Based on the Mantel test and PCA results, **Tree_covered_areas** seems to be the most significant feature for dengue cases in Cambodia.

The accurancy metrics present high mean squared error values which indicate that the algorithem Is not very accurate.   
after reciving those results, I have tried to remove some outliers of points that are far from the standart devision but without any meaningful secsuss.

The reasons to the unscsesful prediction accurancy may be caused by:   
\-The fact that each country has not much values after removing Nan’s, not anogh data for prediction.

\-Not all featurs has info for long years time which may effect the correlation between their values and the dengue cases.

\-The data from country like cmbodia may be unrealibale since the medicine documentation developed lately.

\-Maybe different approach for value normalization is needed for each feature.   
\-each feature from the IMF DB has a lot of tables. I have not used all of them. Considering searching for another DB that contains the climate change parameters in all industries.

Next steps for this project:

\-I would try to ask another question- if the dengue fever can be a pandemic sometime in the future based on the climate change parameter (of course I would need to find some better correlated features\\data.)
