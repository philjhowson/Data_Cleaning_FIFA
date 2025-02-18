import streamlit as st

def cleaning():
    
    st.markdown("""<h2 style="font-size: 1.5em;"><u>
                Data Cleaning</u></h2>""", unsafe_allow_html = True)

    
    st.markdown("""I found the dataset to have various inconistencies
                in how data was labeled, the units that were used, date
                formats, and other extraneous symbols.<br>
                The first thing I did was examine for NaN values, for which
                I found only one column, 'Hits' with NaNs. I converted these
                values to 0 because they appear to represent the absence of
                hits, rather than an unknown value. Thus making NaN a true
                zero in this case.<br>
                The 'Hits', 'Value' 'Height', and 'Weight' columns additionally
                had following unit measurements (e.g., 'cm') which were not always
                consistent (e.g., sometimes height was in 'cm' and sometimes
                they were in feet and inches). Custom functions were written
                to convert height and weight to the same units. For 'Value', I
                also converted the units to millions, and Wage was converted
                into thousands. Finally, the columns such as 'CM' had a ★ which
                needed to be removed for conversion into numeric values and Club
                had extraneous \n values, which had to be removed.""",
                unsafe_allow_html = True)
