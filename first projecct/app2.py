import streamlit as st

import numpy as np
import pandas as pd
import joblib

model = joblib.load('finalized_model.sav')
scaler = joblib.load('std_scaler.bin')

st.title('How much is the house worth?')
# PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
Area = st.number_input("Input House Area", 0,300)
Room = st.selectbox("How many room it has", [1,2,3,4,5])
Parking = st.selectbox("Has Parking?", [0 , 1])
Warehouse = st.selectbox("Has Warehouse?", [0 , 1])
Elevator = st.selectbox("Has Elevator?", [0 , 1])
Address = st.selectbox("What is its address?", ['Abazar', 'Abbasabad', 'Abuzar', 'Afsarieh', 'Ahang', 'Air force', 'Ajudaniye', 'Alborz Complex', 'Aliabad South', 'Amir Bahador',
        'Amirabad', 'Amirieh', 'Andisheh', 'Aqdasieh', 'Araj', 'Atabak', 'Azadshahr', 'Azarbaijan', 'Azari', 'Baghestan', 'Bahar', 
        'Baqershahr', 'Beryanak', 'Boloorsazi', 'Central Janatabad', 'Chahardangeh', 'Chardangeh', 'Chardivari', 'Chidz', 'Damavand', 
        'Darabad', 'Darakeh', 'Darband', 'Daryan No', 'Dehkade Olampic', 'Dezashib', 'Dolatabad', 'Dorous', 'East Ferdows Boulevard', 
        'East Pars', 'Ekbatan', 'Ekhtiarieh', 'Elahieh', 'Elm-o-Sanat', 'Enghelab', 'Eram', 'Eskandari', 'Fallah', 'Farmanieh', 'Fatemi', 
        'Feiz Garden', 'Firoozkooh', 'Firoozkooh Kuhsar', 'Garden of Saba', 'Gheitarieh', 'Ghiyamdasht', 'Ghoba', 'Gholhak', 'Gisha', 
        'Golestan', 'Haft Tir', 'Hakimiyeh', 'Hashemi', 'Hassan Abad', 'Hekmat', 'Heravi', 'Heshmatieh', 'Hor Square', 'Islamshahr', 
        'Islamshahr Elahieh', 'Javadiyeh', 'Jeyhoon', 'Jordan', 'Kahrizak', 'Kamranieh', 'Karimkhan', 'Karoon', 'Kazemabad', 
        'Keshavarz Boulevard', 'Khademabad Garden', 'Khavaran', 'Komeil', 'Koohsar', 'Kook', 'Lavizan', 'Mahallati', 'Mahmoudieh', 
        'Majidieh', 'Malard', 'Marzdaran', 'Mehrabad', 'Mehrabad River River', 'Mehran', 'Mirdamad', 'Mirza Shirazi', 'Moniriyeh', 
        'Narmak', 'Nasim Shahr', 'Nawab', 'Naziabad', 'Nezamabad', 'Niavaran', 'North Program Organization', 'Northern Chitgar', 
        'Northern Janatabad', 'Northern Suhrawardi', 'Northren Jamalzadeh', 'Ostad Moein', 'Ozgol', 'Pakdasht', 'Pakdasht KhatunAbad', 
        'Parand', 'Parastar', 'Pardis', 'Pasdaran', 'Persian Gulf Martyrs Lake', 'Pirouzi', 'Pishva', 'Punak', 'Qalandari', 'Qarchak', 
        'Qasr-od-Dasht', 'Qazvin Imamzadeh Hassan', 'Railway', 'Ray', 'Ray - Montazeri', 'Ray - Pilgosh', 'Razi', 'Republic', 'Robat Karim',
        'Rudhen', 'Saadat Abad', 'SabaShahr', 'Sabalan', 'Sadeghieh', 'Safadasht', 'Salehabad', 'Salsabil', 'Sattarkhan', 'Seyed Khandan',
        'Shadabad', 'Shahedshahr', 'Shahr-e-Ziba', 'ShahrAra', 'Shahrake Apadana', 'Shahrake Azadi', 'Shahrake Gharb', 'Shahrake Madaen', 
        'Shahrake Qods', 'Shahrake Quds', 'Shahrake Shahid Bagheri', 'Shahrakeh Naft', 'Shahran', 'Shahryar', 'Shams Abad', 'Shoosh', 
        'Si Metri Ji', 'Sohanak', 'Southern Chitgar', 'Southern Janatabad', 'Southern Program Organization', 'Southern Suhrawardi',
        'Tajrish', 'Tarasht', 'Taslihat', 'Tehran Now', 'Tehransar', 'Telecommunication', 'Tenant', 'Thirteen November', 'Vahidieh', 
        'Vahidiyeh', 'Valiasr', 'Vanak', 'Velenjak', 'Villa', 'Water Organization', 'Waterfall', 'West Ferdows Boulevard', 'West Pars', 
        'Yaftabad', 'Yakhchiabad', 'Yousef Abad', 'Zafar', 'Zaferanieh', 'Zargandeh', 'Zibadasht'])

def predict(): 
    row = np.array([Area,Room,Parking,Warehouse,Elevator,Address]) 
    X = pd.DataFrame([row], columns = ['Area', 'Room', 'Parking', 'Warehouse', 'Elevator', 'Address'])

    X['Area'] = scaler.fit_transform(X[['Area']])

    X[['Abazar', 'Abbasabad', 'Abuzar', 'Afsarieh', 'Ahang', 'Air force', 'Ajudaniye', 'Alborz Complex', 'Aliabad South', 'Amir Bahador',
        'Amirabad', 'Amirieh', 'Andisheh', 'Aqdasieh', 'Araj', 'Atabak', 'Azadshahr', 'Azarbaijan', 'Azari', 'Baghestan', 'Bahar', 
        'Baqershahr', 'Beryanak', 'Boloorsazi', 'Central Janatabad', 'Chahardangeh', 'Chardangeh', 'Chardivari', 'Chidz', 'Damavand', 
        'Darabad', 'Darakeh', 'Darband', 'Daryan No', 'Dehkade Olampic', 'Dezashib', 'Dolatabad', 'Dorous', 'East Ferdows Boulevard', 
        'East Pars', 'Ekbatan', 'Ekhtiarieh', 'Elahieh', 'Elm-o-Sanat', 'Enghelab', 'Eram', 'Eskandari', 'Fallah', 'Farmanieh', 'Fatemi', 
        'Feiz Garden', 'Firoozkooh', 'Firoozkooh Kuhsar', 'Garden of Saba', 'Gheitarieh', 'Ghiyamdasht', 'Ghoba', 'Gholhak', 'Gisha', 
        'Golestan', 'Haft Tir', 'Hakimiyeh', 'Hashemi', 'Hassan Abad', 'Hekmat', 'Heravi', 'Heshmatieh', 'Hor Square', 'Islamshahr', 
        'Islamshahr Elahieh', 'Javadiyeh', 'Jeyhoon', 'Jordan', 'Kahrizak', 'Kamranieh', 'Karimkhan', 'Karoon', 'Kazemabad', 
        'Keshavarz Boulevard', 'Khademabad Garden', 'Khavaran', 'Komeil', 'Koohsar', 'Kook', 'Lavizan', 'Mahallati', 'Mahmoudieh', 
        'Majidieh', 'Malard', 'Marzdaran', 'Mehrabad', 'Mehrabad River River', 'Mehran', 'Mirdamad', 'Mirza Shirazi', 'Moniriyeh', 
        'Narmak', 'Nasim Shahr', 'Nawab', 'Naziabad', 'Nezamabad', 'Niavaran', 'North Program Organization', 'Northern Chitgar', 
        'Northern Janatabad', 'Northern Suhrawardi', 'Northren Jamalzadeh', 'Ostad Moein', 'Ozgol', 'Pakdasht', 'Pakdasht KhatunAbad', 
        'Parand', 'Parastar', 'Pardis', 'Pasdaran', 'Persian Gulf Martyrs Lake', 'Pirouzi', 'Pishva', 'Punak', 'Qalandari', 'Qarchak', 
        'Qasr-od-Dasht', 'Qazvin Imamzadeh Hassan', 'Railway', 'Ray', 'Ray - Montazeri', 'Ray - Pilgosh', 'Razi', 'Republic', 'Robat Karim',
        'Rudhen', 'Saadat Abad', 'SabaShahr', 'Sabalan', 'Sadeghieh', 'Safadasht', 'Salehabad', 'Salsabil', 'Sattarkhan', 'Seyed Khandan',
        'Shadabad', 'Shahedshahr', 'Shahr-e-Ziba', 'ShahrAra', 'Shahrake Apadana', 'Shahrake Azadi', 'Shahrake Gharb', 'Shahrake Madaen', 
        'Shahrake Qods', 'Shahrake Quds', 'Shahrake Shahid Bagheri', 'Shahrakeh Naft', 'Shahran', 'Shahryar', 'Shams Abad', 'Shoosh', 
        'Si Metri Ji', 'Sohanak', 'Southern Chitgar', 'Southern Janatabad', 'Southern Program Organization', 'Southern Suhrawardi',
        'Tajrish', 'Tarasht', 'Taslihat', 'Tehran Now', 'Tehransar', 'Telecommunication', 'Tenant', 'Thirteen November', 'Vahidieh', 
        'Vahidiyeh', 'Valiasr', 'Vanak', 'Velenjak', 'Villa', 'Water Organization', 'Waterfall', 'West Ferdows Boulevard', 'West Pars', 
        'Yaftabad', 'Yakhchiabad', 'Yousef Abad', 'Zafar', 'Zaferanieh', 'Zargandeh', 'Zibadasht']] = False

    X.at[0, X['Address'][0]] = True
    X = X.drop('Address' , axis = 1)

    prediction = model.predict(X)
    
    st.success(f'your house price is: {int(prediction[0]):,}' )

trigger = st.button('Predict', on_click=predict)