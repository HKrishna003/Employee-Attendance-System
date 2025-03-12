import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import plotly.express as px

excel_file = pd.ExcelFile("T1.xlsx")

images = r"D:\srec_project\project\backend_1\Dataset_5"

# Get all sheet names
sheet_names = excel_file.sheet_names
# print(sheet_names)
# df = pd.read_excel("T1.xlsx", sheet_name="Sheet2")
# print(df)


# Function to convert time to 12-hour format (only extracting hours)
def convert_to_12_hour(time_str):
    dt = datetime.strptime(time_str, "%a %b %d %H:%M:%S %Y")
    return dt.strftime("%I").lstrip("0")  # Extracts hour and removes leading zero

st.set_page_config(
    page_title="Employee Attendance Dashboard",
    page_icon="🧑‍🏭",
    layout="wide",
)

with st.sidebar:
    st.title('🧑‍🏭 Employee Attendance Dashboard 🧑‍🏭')
    selected_dept = st.selectbox('Select Department', sheet_names)
    month = pd.read_excel("T1.xlsx",sheet_name=selected_dept)
    employee_names = month['Name'].tolist()
    selected_employee = st.selectbox('Select an employee', employee_names)
    
col1, col2 = st.columns(2)

with col1:
    st.title(selected_employee)
    person_image_1 = images + "\\" + selected_dept + "\\" + selected_employee + "\\" + selected_employee+"_1"+".jpg"
    print(person_image_1)
    st.image(person_image_1, caption=f"Employee ID : {id}", use_container_width=True)
    
    in_time = month.loc[month["Name"] == selected_employee, "In"].values
    out_time = month.loc[month["Name"] == selected_employee, "Out"].values


    in_hours = [convert_to_12_hour(x.strip()) for x in str(in_time[0]).split(",") if x.strip()]
    out_hours = [convert_to_12_hour(x.strip()) for x in str(out_time[0]).split(",") if x.strip()]
    
    # chart_data = pd.DataFrame({
    # "In_Hour": in_hours,
    # "Out_Hour": out_hours
    # })
    print(in_hours)
    print(out_hours)
    # st.line_chart(chart_data)
    in_hours = list(map(int, in_hours))
    out_hours = list(map(int, out_hours))

    # Create a DataFrame for Plotly
    df_chart = pd.DataFrame({
        "Hour": list(range(1, len(in_hours) + 1)),  # X-axis (index of timestamps)
        "In_Hour": in_hours,
        "Out_Hour": out_hours
    })

    # Plotly Line Chart with Custom Colors
    fig = px.line(df_chart, x="Hour", y=["In_Hour", "Out_Hour"],
                  labels={"value": "Hour", "variable": "Type"},
                  title="Employee In & Out Times",
                  color_discrete_map={"In_Hour": "red", "Out_Hour": "green"})  # 🎨 Custom Colors

    # Display the chart in Streamlit
    st.plotly_chart(fig)
    
with col2:
    st.write(f'## Attendance Summary {selected_dept}')
    work = month.loc[month["Name"] == selected_employee, "Work"].values[0]
    ot = month.loc[month["Name"] == selected_employee, "OT"].values[0]
    st.metric(label="Work", value=work, delta="Hrs", delta_color="normal")
    st.metric(label="OT", value=ot, delta="Hrs")
    print(work)
