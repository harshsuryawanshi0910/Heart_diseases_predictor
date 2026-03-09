import streamlit as st
import numpy as np
import pandas as pd
import pickle
import sqlite3
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import red, green

st.set_page_config(page_title="Heart Disease Risk Assessment", layout="wide")

# ---------------- DATABASE ----------------

conn = sqlite3.connect("patients.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS patients(
id INTEGER PRIMARY KEY AUTOINCREMENT,
patient_id TEXT,
name TEXT,
age INTEGER,
gender TEXT,
chest_pain TEXT,
exercise_angina TEXT,
max_hr INTEGER,
oldpeak REAL,
st_slope TEXT,
fasting_bs TEXT,
result TEXT,
date TEXT
)
""")

conn.commit()

# ---------------- AUTO PATIENT ID ----------------

def get_next_patient_id():

    row = cursor.execute(
        "SELECT patient_id FROM patients ORDER BY id DESC LIMIT 1"
    ).fetchone()

    if row is None:
        return "PAT-0001"

    last_id = int(row[0].split("-")[1])

    return f"PAT-{last_id+1:04d}"

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():

    try:
        with open("heart_disease_model.pkl","rb") as f:
            obj = pickle.load(f)

        return obj["model"],obj["scaler"],obj["features"]

    except:
        return None,None,None


model,scaler,feature_order=load_model()

# ---------------- PREDICTION ----------------

def predict_risk(age,sex,chest_pain,blood_sugar,max_hr,exercise_angina,oldpeak,st_slope):

    user_input={
        'age':age,
        'sex':sex,
        'chest pain type':chest_pain,
        'fasting blood sugar':blood_sugar,
        'max heart rate':max_hr,
        'exercise angina':exercise_angina,
        'oldpeak':oldpeak,
        'ST slope':st_slope
    }

    ordered=[user_input[f] for f in feature_order]

    features=np.array([ordered]).astype(float)

    if scaler:
        features=scaler.transform(features)

    pred=model.predict(features)[0]

    prob=None

    try:
        prob=model.predict_proba(features)[0]
    except:
        pass

    return pred,prob

# ---------------- PDF REPORT ----------------

def create_pdf(data,result,recommendations,prob):

    filename=f"{data['Patient ID']}_report.pdf"

    c=canvas.Canvas(filename,pagesize=letter)

    y=760

    # Title
    c.setFont("Helvetica-Bold",20)
    c.drawCentredString(300,y,"HEART DISEASE MEDICAL REPORT")

    y-=40

    c.setFont("Helvetica",10)
    c.drawString(40,y,f"Report Generated: {datetime.now().strftime('%d %b %Y')}")

    y-=30

    # Patient Details
    c.setFont("Helvetica-Bold",14)
    c.drawString(40,y,"Patient Details")

    y-=20
    c.setFont("Helvetica",12)

    for k in ["Patient ID","Name","Age","Gender"]:
        c.drawString(60,y,f"{k}: {data[k]}")
        y-=18

    y-=10

    # Clinical Inputs
    c.setFont("Helvetica-Bold",14)
    c.drawString(40,y,"Clinical Measurements")

    y-=20
    c.setFont("Helvetica",12)

    inputs=[
        "Chest Pain Type",
        "Exercise Angina",
        "Max Heart Rate",
        "ST Depression",
        "ST Slope",
        "Fasting Blood Sugar"
    ]

    for i in inputs:
        c.drawString(60,y,f"{i}: {data[i]}")
        y-=18

    y-=20

    # Result
    c.setFont("Helvetica-Bold",16)

    if result=="High Risk":
        c.setFillColor(red)
    else:
        c.setFillColor(green)

    c.drawString(40,y,f"Prediction Result: {result}")

    c.setFillColor("black")

    y-=20

    if prob is not None:

        c.setFont("Helvetica",12)
        c.drawString(60,y,f"Disease Probability: {round(prob[1]*100,2)} %")
        y-=18

        c.drawString(60,y,f"No Disease Probability: {round(prob[0]*100,2)} %")

    y-=25

    # Recommendations
    c.setFont("Helvetica-Bold",14)
    c.drawString(40,y,"Recommendations")

    y-=20
    c.setFont("Helvetica",11)

    for r in recommendations:
        c.drawString(60,y,f"• {r}")
        y-=16

    y-=30

    # Disclaimer
    c.setFont("Helvetica-Oblique",9)
    c.drawString(40,y,"Disclaimer: This AI prediction is for educational purposes only.")
    c.drawString(40,y-12,"Consult a qualified cardiologist for medical advice.")

    c.save()

    return filename

# ---------------- UI ----------------

st.title(" Heart Disease Risk Assessment")

menu=st.sidebar.selectbox("Menu",["Prediction","Admin Dashboard"])

# ---------------- PREDICTION PAGE ----------------

if menu=="Prediction":

    patient_id=get_next_patient_id()

    col1,col2=st.columns(2)

    with col1:
        name=st.text_input("Patient Name")

    with col2:
        st.text_input("Patient ID",value=patient_id,disabled=True)

    st.subheader("Medical Information")

    col1,col2=st.columns(2)

    with col1:

        age=st.number_input("Age",18,100,40)

        sex=st.radio("Gender",["Male","Female"])
        sex_val=1 if sex=="Male" else 0

        chest=st.selectbox(
            "Chest Pain Type",
            ["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"]
        )

        chest_val=["Typical Angina","Atypical Angina","Non-anginal Pain","Asymptomatic"].index(chest)

        angina=st.radio("Exercise Angina",["No","Yes"])
        angina_val=1 if angina=="Yes" else 0

    with col2:

        max_hr=st.slider("Maximum Heart Rate",60,220,150)

        oldpeak=st.slider("ST Depression",-2.0,6.0,1.0)

        slope=st.selectbox("ST Slope",["Upsloping","Flat","Downsloping"])
        slope_val=["Upsloping","Flat","Downsloping"].index(slope)

        fasting=st.radio("Fasting Blood Sugar",["Normal","High"])
        fasting_val=0 if fasting=="Normal" else 1

    if st.button("Assess Heart Disease Risk"):

        pred,prob=predict_risk(
        age,sex_val,chest_val,fasting_val,
        max_hr,angina_val,oldpeak,slope_val
        )

        result="High Risk" if pred==1 else "Low Risk"

        if result=="High Risk":
            st.error("High Risk of Heart Disease")
        else:
            st.success("Low Risk of Heart Disease")

        recommendations=[
        "Consult cardiologist",
        "Maintain healthy lifestyle",
        "Regular health checkups"
        ]

        # SAVE DATA
        cursor.execute("""
        INSERT INTO patients
        (patient_id,name,age,gender,chest_pain,exercise_angina,max_hr,oldpeak,st_slope,fasting_bs,result,date)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
        patient_id,name,age,sex,
        chest,angina,max_hr,oldpeak,
        slope,fasting,result,str(datetime.now())
        ))

        conn.commit()

        st.success("Patient record saved")

        # REPORT PREVIEW

        report_data={
        "Patient ID":patient_id,
        "Name":name,
        "Age":age,
        "Gender":sex,
        "Chest Pain Type":chest,
        "Exercise Angina":angina,
        "Max Heart Rate":max_hr,
        "ST Depression":oldpeak,
        "ST Slope":slope,
        "Fasting Blood Sugar":fasting
        }

        df=pd.DataFrame(list(report_data.items()),columns=["Field","Value"])
        st.table(df)

        pdf=create_pdf(report_data,result,recommendations,prob)

        with open(pdf,"rb") as f:
            st.download_button("Download Medical Report",f,file_name=pdf)

# ---------------- ADMIN DASHBOARD ----------------

if menu=="Admin Dashboard":

    st.title(" Admin Dashboard")

    data=cursor.execute("SELECT * FROM patients").fetchall()

    columns=[
    "DB_ID","Patient ID","Name","Age","Gender",
    "Chest Pain","Exercise Angina","Max HR",
    "ST Depression","ST Slope","Fasting BS",
    "Result","Date"
    ]

    df=pd.DataFrame(data,columns=columns)

    st.metric("Total Patients",len(df))

    if len(df)>0:

        display_df=df.drop(columns=["DB_ID"])

        st.dataframe(display_df,use_container_width=True)

    else:

        st.warning("No patient records found")