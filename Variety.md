# BarChart with Error Bars
![](All_screenshots.jpg)

Python Code:
```
import matplotlib.pyplot as plt
import numpy as np

#gal-H2O kWh−1
water=[6.05,.74,.52,.49,.04,.001,.001,.001,.0001]
errorbars=[1.55,0,.15,0,0,0,0,0,0]

fuels=["Hydro", "CSP*", "Nuclear*", "Coal-CCS", "Solar-PV", "Geothermal", "Wind", "Wave", "Tidal"]

index=np.arange(len(fuels)) + 0.3
print(len(water),len(errorbars),len(index))

#plt.figure()#figsize=(15, 10))
#plt.bar(index,water)
#plt.errorbar(index,errorbars)

fig, ax = plt.subplots(figsize=(15, 10))
ax.bar(index, water, yerr=errorbars, align='center', alpha=0.5, ecolor='gray', capsize=10)
ax.set_xticks(index)
ax.set_xticklabels(fuels, Fontsize=15)
plt.yticks(fontsize=20)
plt.ylabel("Gallons of Water per kWh", fontsize=25)
plt.title("Water Needs of Various Electricity Sources", fontsize=25)

rects = ax.patches
labels = [f"{i}" for i in water]

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(
        rect.get_x() + rect.get_width() / 2, height/2, label, ha="center", va="bottom", fontsize=15
    )
```

# Data Table, Scatterplot- Transparent & Various Sizes, Scatterplot - Label Points

Python Code:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This makes out plots higher resolution, which makes them easier to see while building

df = pd.read_csv('OECD 2019.csv')
df.head()

###### Break Here in Jupyter ##########
ghg=df['GHG']
mat=df['Material']
gdp=df['GDP']

area=gdp/1000 * 2
plt.scatter(ghg, mat, s=area, alpha=.5)

plt.xlabel('Total Greenhouse Gases (Excl. LULUCF) per Capita')
plt.ylabel('Matrial Consumption per Capita')

###### Break Here in Jupyter ##########
fig, ax = plt.subplots(figsize=(10,10))

scatter = ax.scatter(ghg, mat,color='#e8634a', s=area, alpha=.5)

area = (gdp/10000)**3

# produce a legend with a cross section of sizes from the scatter
handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)

labels_s=14

plt.text(x=df.GHG[df.CountryCode=='LUX']+0.7,y=df.Material[df.CountryCode=='LUX']+0.7,s='LUX',fontsize=labels_s)
plt.text(x=df.GHG[df.CountryCode=='IRL']+0.45,y=df.Material[df.CountryCode=='IRL']+0.45,s='IRL',fontsize=labels_s)
plt.text(x=df.GHG[df.CountryCode=='USA']+0.35,y=df.Material[df.CountryCode=='USA']+0.35,s='USA',fontsize=labels_s)
plt.text(x=df.GHG[df.CountryCode=='CAN']+0.3,y=df.Material[df.CountryCode=='CAN']+0.3,s='CAN',fontsize=labels_s)
plt.text(x=df.GHG[df.CountryCode=='CHE']-1.3,y=df.Material[df.CountryCode=='CHE']+0.3, s='CHE',fontsize=labels_s)
plt.text(x=df.GHG[df.CountryCode=='NLD']+0.3,y=df.Material[df.CountryCode=='NLD']+0.3,s='NLD',fontsize=labels_s)
plt.text(x=df.GHG[df.CountryCode=='FIN']+0.3,y=df.Material[df.CountryCode=='FIN']+0.3,s='FIN',fontsize=labels_s)
plt.text(x=df.GHG[df.CountryCode=='NOR']-1.3,y=df.Material[df.CountryCode=='NOR']+0.3,s='NOR',fontsize=labels_s)
plt.text(x=df.GHG[df.CountryCode=='SWE']+0.3,y=df.Material[df.CountryCode=='SWE']+0.3,s='SWE',fontsize=labels_s)
plt.text(x=df.GHG[df.CountryCode=='DNK']+0.3,y=df.Material[df.CountryCode=='DNK']+0.3,s='DNK',fontsize=labels_s)
plt.text(x=df.GHG[df.CountryCode=='AUS']+0.3,y=df.Material[df.CountryCode=='AUS']+0.3,s='AUS',fontsize=labels_s)
plt.text(x=df.GHG[df.CountryCode=='NZL']+0.3,y=df.Material[df.CountryCode=='NZL']+0.3,s='NZL',fontsize=labels_s)
plt.text(x=df.GHG[df.CountryCode=='ISL']+0.3,y=df.Material[df.CountryCode=='ISL']+0.3,s='ISL',fontsize=labels_s)
plt.text(x=df.GHG[df.CountryCode=='JPN']+0.25,y=df.Material[df.CountryCode=='JPN']+0.25,s='JPN',fontsize=labels_s)
plt.text(x=df.GHG[df.CountryCode=='OECD']+0.25,y=df.Material[df.CountryCode=='OECD']+0.25,s='OEDC-All',fontsize=labels_s)
plt.text(x=df.GHG[df.CountryCode=='EST']+0.25,y=df.Material[df.CountryCode=='EST']+0.25,s='EST',fontsize=labels_s)
plt.text(x=df.GHG[df.CountryCode=='RUS']+0.2,y=df.Material[df.CountryCode=='RUS']+0.2,s='RUS',fontsize=labels_s)
#plt.text(x=df.GHG[df.CountryCode=='OECDE']+0.3,y=df.Material[df.CountryCode=='OECDE']+0.3,s='OEDC-Europe')

plt.xlabel('Total Greenhouse Gases (Excl. LULUCF) per Capita', fontsize=20)
plt.ylabel('Material Consumption per Capita',rotation='horizontal', fontsize=20)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

plt.xticks(fontsize=labels_s)
plt.yticks(fontsize=labels_s)

#legend2 = ax.legend(handles, labels, loc="upper right", title="Sizes")
```

# Chord Diagram
R Code:
```
install.packages("hrbrthemes")
devtools::install_github("mattflor/chorddiag")
install.packages("chorddiag")


# Libraries
library(tidyverse)
library(viridis)
library(patchwork)
library(hrbrthemes)
library(circlize)
library(chorddiag)  #devtools::install_github("mattflor/chorddiag")

# Load dataset from github
#data <- read.table("/Users/dglenn/Desktop/ChordDiagram/MinJanTempRound5-0509.csv", header=TRUE)
#data <- read.table("/Users/dglenn/Desktop/ChordDiagram/MinJanTempRound5.csv", header=TRUE)
#data <- read.table("/Users/dglenn/Desktop/ChordDiagram/WhereColdGoes.csv", header=TRUE)
data <- read.table("/Users/dglenn/Desktop/ChordDiagram/MinJanTempBins.csv", header=TRUE)
#data <- read.table("/Users/dglenn/Desktop/ChordDiagram/MinJanTempBins-0509.csv", header=TRUE)
#numcat<-17 #number of categories
#numcat<-15 #number of categories
numcat<-5 #number of categories

# short names

#colnames(data) <- c('-20','-15','-10','-5',	'0', '5', '10', '15', '20', '25', '30',	'35',	'40',	'45',	'50',	'55',	'60')
#colnames(data) <- c('-10',	'-5',	'0', '5', '10', '15', '20', '25', '30',	'35',	'40',	'45',	'50',	'55',	'60')
colnames(data) <- c("<10","10s","20s","30s","40+")
rownames(data) <- colnames(data)

# I need a long format
data_long <- data %>%
  rownames_to_column %>%
  gather(key = 'key', value = 'value', -rowname)

# parameters
circos.clear()
circos.par(start.degree = 90, gap.degree = 4, track.margin = c(-0.1, 0.1), points.overflow.warning = FALSE)
par(mar = rep(0, 4))

# color palette
mycolor <- viridis(numcat, alpha = 1, begin = 0, end = 1, option = "D")
mycolor <- mycolor[sample(1:numcat)]
mycolor <- c("#481567FF","#2D708EFF","#29AF7FFF","#95D840FF","#FDE725FF")
mycolor <- c("#0066cc","#339966","#FDE725FF","#ffa64d","#ff0000")

# Base plot
chordDiagram(
  x = data_long, 
  grid.col = mycolor,
  transparency = 0.25,
  directional = 1,
  direction.type = c("arrows", "diffHeight"), 
  diffHeight  = -0.04,
  annotationTrack = "grid", 
  annotationTrackHeight = c(0.05, 0.1),
  link.arr.type = "big.arrow", 
  link.sort = TRUE, 
  link.largest.ontop = TRUE)

# Add text and axis
circos.trackPlotRegion(
  track.index = 1, 
  bg.border = NA, 
  panel.fun = function(x, y) {
    
    xlim = get.cell.meta.data("xlim")
    sector.index = get.cell.meta.data("sector.index")
    
    # Add names to the sector. 
    circos.text(
      x = mean(xlim), 
      y = 3.2, 
      labels = sector.index, 
      facing = "bending", 
      cex = 0.8
    )
    
    # Add graduation on axis
    #circos.axis(
      #h = "top", 
      #major.at = seq(from = 0, to = xlim[2], by = ifelse(test = xlim[2]>10, yes = 2, no = 1)), 
      #minor.ticks = 1, 
      #major.tick.percentage = 0.5,
      #labels.niceFacing = FALSE)
  }
)
```

# Line Graph with Propbalitiy Cones and Value Point data

```
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mtick

df= pd.read_csv('2oC-10years-global.csv')

year= df['Year']
biomass=df['Biomass']*100
biomass_top=df['Biomass- top']*100
biomass_bot=df['Biomass- bottom']*100
coal=df['Coal']*100
coal_top=df['Coal- top']*100
coal_bot=df['Coal- bottom']*100
gas=df['Gas']*100
gas_top=df['Gas- top']*100
gas_bot=df['Gas- bottom']*100
hydro=df['Hydro']*100
hydro_top=df['Hydro- top']*100
hydro_bot=df['Hydro- bottom']*100
windSol=df['Wind & Solar (Geothermal & Ocean)']*100
windSol_top=df['Wind & Solar (Geothermal & Ocean)- top']*100
windSol_bot=df['Wind & Solar (Geothermal & Ocean)- bottom']*100
nuclear=df['Nuclear']*100
nuclear_top=df['Nuclear- top']*100
nuclear_bot=df['Nuclear- bottom']*100
oil=df['Oil']*100
oil_top=df['Oil- top']*100
oil_bot=df['Oil- bottom']*100

plt.figure(figsize=(15, 10))
sns.set(font_scale = 2)
sns.set_style("whitegrid")
#sns.set_height(5)

ax = sns.lineplot(year, coal, color='firebrick') 
ax.fill_between(year, coal_bot, coal_top, alpha=0.3, color='gray'); 
ax = sns.lineplot(year, gas, color='darkorange') 
ax.fill_between(year, gas_bot, gas_top, alpha=0.3, color='gray'); 
ax = sns.lineplot(year, nuclear, color='darkviolet') 
ax.fill_between(year, nuclear_bot, nuclear_top, alpha=0.3, color='darkviolet');
ax = sns.lineplot(year, hydro, color='royalblue') 
ax.fill_between(year, hydro_bot, hydro_top, alpha=0.3, color='gray'); 
ax = sns.lineplot(year, oil, color='k')
ax.fill_between(year, oil_bot, oil_top, alpha=0.3, color='gray'); 
ax = sns.lineplot(year, biomass, color='peru')
ax.fill_between(year, biomass_bot, biomass_top, alpha=0.3, color='gray'); 
ax = sns.lineplot(year, windSol, color='g')
ax.fill_between(year, windSol_bot, windSol_top, alpha=0.3, color= 'gray');

ax = sns.lineplot(year, nuclear, color='darkviolet') 
ax.fill_between(year, nuclear_bot, nuclear_top, alpha=0.3, color='darkviolet');
ax = sns.lineplot(year, windSol, color='g')
ax.fill_between(year, windSol_bot, windSol_top, alpha=0.3, color= 'g');

plt.scatter([2020],[3], color= 'k', s=10**2)
plt.scatter([2020],[10], color= 'darkviolet', s=10**2)
plt.scatter([2020],[16], color= 'royalblue', s=10**2)
plt.scatter([2020],[23], color= 'darkorange', s=10**2)
plt.scatter([2020],[35], color= 'firebrick', s=10**2)
plt.scatter([2020],[12], color= 'g', s=10**2)

ax.set_ylabel('Electricity Mix %')
fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
yticks = mtick.FormatStrFormatter(fmt)
ax.yaxis.set_major_formatter(yticks)
ax.set_xticks([2010,2020,2030,2040,2050])
#plt.legend(labels=["Oil","Biomass","Coal","Gas","Hydro","Nuclear","Wind/Solar (and Geo/Ocean)"], fontsize = 20)

print("2oC-10years-Global")
print("55 Models")
```


# Random Forest Variable Testing

Python code: 
```
###This is the one to use
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import glob
import datetime

df = pd.read_csv('ES_randforest-Frijoles.csv')

ID= df['ID']
#y = df['IntMigration_NetRate'] 
y = df['IntMigration_Zscore'] 

feature_names = ['PopDensity','HomicideRate',
                 'NoPobres-1yb','PovertyExtreme-1yb','PovertyRelative-1yb',
                 'Precip-Zscore','Precip-Zscore-1yb','Precip-Zscore-2yb',
                 'FrijolQQ-Zscore']

feature_names = ['Population','PopDensity','HomicideRate','HomicideRate-1yb',
                 'NoPobres-1yb','PovertyExtreme-1yb','PovertyRelative-1yb',
                 'NoPobres-2yb','PovertyExtreme-2yb','PovertyRelative-2yb',
                 'NoPobres-3yb','PovertyExtreme-3yb','PovertyRelative-3yb',
                 'Precip-Zscore','Precip-Zscore-1yb','Precip-Zscore-2yb','FrijolQQ-Zscore',]

feature_names = ['PopDensity','HomicideRate',
                 'NoPobres-1yb','PovertyExtreme-1yb',
                 'NoPobres-2yb','PovertyExtreme-2yb','PovertyRelative-2yb',
                 'PovertyExtreme-3yb','PovertyRelative-3yb',
                 'Precip-Zscore','Precip-Zscore-1yb','Precip-Zscore-2yb','FrijolQQ-Zscore',]


feature_names = ['PopDensity','HomicideRate','PovertyExtreme-1yb','PovertyExtreme-2yb',
                 'Precip-Zscore','Precip-Zscore-1yb','FrijolQQ-Zscore',]

##################################################
from sklearn.model_selection import train_test_split
X = df[feature_names]

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

#Import Random Forest Model
from sklearn.ensemble import RandomForestRegressor

#Create a Regressor
reg=RandomForestRegressor(n_estimators=5000, max_features="sqrt", random_state = 42)

#Train the model using the training sets y_pred=reg.predict(X_test)
reg.fit(X_train,y_train)

y_pred=reg.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.r2_score(y_test, y_pred))

importances = reg.feature_importances_
std = np.std([tree.feature_importances_ for tree in reg.estimators_], axis=0)

forest_importances = pd.Series(importances, index=feature_names)

print("Variable Allocation:")
print(forest_importances.sort_values(ascending=False))

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,7))

forest_importances.plot.bar(yerr=std, ax=ax1,fontsize = 14.0)
ax1.set_title("Feature importances using MDI",fontsize = 16.0)
ax1.set_ylabel("Mean decrease in impurity",fontsize = 16.0)
fig.tight_layout()

ax2.scatter(y_test, y_pred, alpha=.2)
ax2.set_xlabel("Actual",fontsize = 16.0)
ax2.set_ylabel("Predicited",fontsize = 16.0)
ax2.set_title("Accuracy: {}".format(round(metrics.r2_score(y_test, y_pred),3)),fontsize = 16.0)
fig.tight_layout()
```

# Random Forest Variable Explanation

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

df = pd.read_csv('ES_ranforest_homPrecipPoverty.csv')

ID= df['ID']
y = df['IntMigration_NetRate'] 
#y = df['IntMigration_Zscore'] 
feature_names = ['Population','PopDensity','HomicideRate','Precip-Zscore','Precip-Zscore-1yb','Precip-Zscore-2yb',
                 'NoPobres','PovertyExtreme','PovertyRelative',
                 'NoPobres-1yb','PovertyExtreme-1yb','PovertyRelative-1yb',
                 'NoPobres-2yb','PovertyExtreme-2yb','PovertyRelative-2yb',
                 'NoPobres-3yb','PovertyExtreme-3yb','PovertyRelative-3yb']

feature_names = ['PopDensity','HomicideRate','Precip-Zscore','PovertyExtreme-2yb']

X =  df[feature_names]

from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

### Import Random Forest Model
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=5000, max_features="sqrt")
reg.fit(X_train, y_train)

default_importance = reg.feature_importances_

from sklearn.inspection import permutation_importance
perm_importance = permutation_importance(reg, X_train, y_train)

y_pred = reg.predict(X_test)
Y_predict_train = reg.predict(X_train)

from sklearn.metrics import mean_squared_error
mse_train = mean_squared_error(y_train, Y_predict_train)
mse = mean_squared_error(y_test, y_pred)
print(mse_train)
print(mse)

print(reg.score(X_train,y_train))
print(reg.score(X_test,y_test))

from sklearn.inspection import plot_partial_dependence
from sklearn.inspection import partial_dependence

#The partial_dependence function returns the dependencies and the grid
PDs, grid = partial_dependence(reg, X_train, features = ['PopDensity'], percentiles = [0,1])

#The plot_partial_dependence function returns a plot, but can also be unpacked into dependencies and grids
#plot_partial_dependence(reg, X_train, features = ['PopDensity'], percentiles = [0,1]);

def get_PDPvalues(col_name, data, model, grid_resolution = 100):
    Xnew = data.copy()
    sequence = np.linspace(np.min(data[col_name]), np.max(data[col_name]), grid_resolution)
    Y_pdp = []
    for each in sequence:
        Xnew[col_name] = each
        Y_temp = model.predict(Xnew)
        Y_pdp.append(np.mean(Y_temp))
    return pd.DataFrame({col_name: sequence, 'PDs': Y_pdp})

def plot_PDP(col_name, data, model):
    df = get_PDPvalues(col_name, data, model)
    plt.rcParams.update({'font.size': 16})
    plt.rcParams["figure.figsize"] = (6,5)
    fig, ax = plt.subplots()
    ax.plot(data[col_name], np.zeros(data[col_name].shape)+min(df['PDs'])-1, 'k|', ms=15)  # rug plot
    ax.plot(df[col_name], df['PDs'], lw = 2)
    ax.set_ylabel('Partial Dependence')
    return ax

###### Break Here in Jupyter ##########
ax = plot_PDP('PopDensity', X_train_reduced, reg)
ax.set_xlabel('Population Density')
plt.tight_layout();

ax = plot_PDP('HomicideRate', X_train_reduced, reg)
ax.set_xlabel('Homicide Rate')
plt.tight_layout();
```
