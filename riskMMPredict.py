import os
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import re
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

script_dir = os.path.dirname(os.path.abspath(__file__)) 
file_path = os.path.join(script_dir, "weatherAUS.csv")
data = pd.read_csv(file_path)

#přidání okolních měst pro větší přesnost předpovědi
target_city = "Melbourne"
mesta = data['Location'].unique()
geolocator = Nominatim(user_agent="weather_project")
cords = {}
for mesto in mesta:
    cisty_nazev = re.sub(r'([a-z])([A-Z])', r'\1 \2', mesto)
    dotaz = f"{cisty_nazev}, Australia"
    lokace = geolocator.geocode(dotaz)
    if lokace:
        cords[mesto] = (lokace.latitude, lokace.longitude)
    else:
        # Kdyby náhodou letiště stále nenašel, zkusíme explicitní dotaz
        lokace_alt = geolocator.geocode(f"{cisty_nazev} Victoria, Australia") #Victoria protože jsem si vybral Melbourne, který je v regionu Victoria (pojistka pro větší přesnost) 
        if lokace_alt:
            cords[mesto] = (lokace_alt.latitude, lokace_alt.longitude)
    time.sleep(1)
target_cords = cords[target_city]
vzdalenosti = {}
for mesto, bod in cords.items():
    vzdalenosti[mesto]= round(geodesic(target_cords, bod).kilometers,1)
nejblizsi = sorted(vzdalenosti.items(), key= lambda x: x[1])

nejblizsi3 = [ mesto for mesto, dist in nejblizsi[:4]]
vzdalenostti3 = dict(nejblizsi[:4])
filtereddata = data [data['Location'].isin(nejblizsi3) & (data['Location'] != 'Richmond') ].copy() #richmond ne protože nezachytává některá data.
filtereddata['DistanceToMelbourne'] = filtereddata['Location'].map(vzdalenostti3)


cleanData = filtereddata.dropna().copy()
cleanData = cleanData.drop(columns=['RainTomorrow'])
cleanData = cleanData.drop(columns=['RainToday'])

cleanData['Date'] = pd.to_datetime(cleanData['Date'])
cleanData['DayOfYear'] = cleanData['Date'].dt.dayofyear

cleanData['DayOfYear_sin'] = np.sin( np.pi * 2 * cleanData['DayOfYear'] / 365.25 )
cleanData['DayOfYear_cos'] = np.cos( np.pi * 2 * cleanData['DayOfYear'] / 365.25 )

windSmer = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
}
cleanData['WindGustDir_deg'] = cleanData['WindGustDir'].map(windSmer)
cleanData['WindDir9am_deg'] = cleanData['WindDir9am'].map(windSmer)
cleanData['WindDir3pm_deg'] = cleanData['WindDir3pm'].map(windSmer)

cleanData['WindGustDir_sin'] = np.sin(np.radians(cleanData['WindGustDir_deg']))
cleanData['WindGustDir_cos'] = np.cos(np.radians(cleanData['WindGustDir_deg']))

cleanData['WindDir9am_sin'] = np.sin(np.radians(cleanData['WindDir9am_deg']))
cleanData['WindDir9am_cos'] = np.cos(np.radians(cleanData['WindDir9am_deg']))

cleanData['WindDir3pm_sin'] = np.sin(np.radians(cleanData['WindDir3pm_deg']))
cleanData['WindDir3pm_cos'] = np.cos(np.radians(cleanData['WindDir3pm_deg']))
#odstranit nepotřebné
nepotrebneSloupce = ['WindDir9am_deg', 'WindGustDir_deg', 'WindDir3pm_deg', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'Date', 'DayOfYear', 'Location']
cleanData = cleanData.drop(columns=nepotrebneSloupce)

from sklearn.model_selection import KFold, cross_val_predict
X = cleanData.drop(columns=['RISK_MM'])
y = cleanData['RISK_MM']

kf = KFold(n_splits=10, shuffle=True, random_state=64)

# 2. Nastavíme 10-fold validaci pro regresi
# (Zde se nepoužívá Stratified, protože nemáme kategorie 0/1, ale obyčejný KFold)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 3. Model: Min-Max škálování + kNN Regresor (např. s k=25)
knn_regressor = Pipeline([
    ('scaler', MinMaxScaler()),
    ('knn', KNeighborsRegressor(n_neighbors=25))
])


predikce_srazek = cross_val_predict(knn_regressor, X, y, cv=kf)

# Vyhodnocení, jak moc se model v průměru sekne (MAE)
mae = mean_absolute_error(y, predikce_srazek)
print(f"--- Výsledky regrese kNN ---")
print(f"Průměrná chyba předpovědi (MAE): {mae:.2f} mm")

#Ukázka skutečnosti vs. předpovědi pro dny, kdy reálně pršelo
vysledky_df = pd.DataFrame({
    'Skutečné srážky (mm)': y,
    'Předpověď modelu (mm)': predikce_srazek.round(1)
})

# Vypíšeme prvních 10 dní, kdy reálně spadlo aspoň 1 mm vody
print("\nUkázka srovnání pro deštivé dny:")
print(vysledky_df[vysledky_df['Skutečné srážky (mm)'] >= 1.0].head(10))