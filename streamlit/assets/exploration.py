import streamlit as st
from PIL import Image

def exploration():
    
    st.markdown("""<h2 style="font-size: 1.5em;"><u>
                Data Exploration
                & Preparation</u></h2>""", unsafe_allow_html = True)

    st.markdown("""To examine the relationship between features, I
                plotted a scatter plot of different features against
                wage and a correlation matrix. On the first pass, plots
                were genereated features that were strongly correlated
                were pruned. For example, the feature 'Total Stats' was
                strongly correlated with other metrics of skill, so
                those features were removed and I kept 'Total Stats'
                because it represents the overall performance of the
                player. I did this to avoid multicollinearity and
                make the final models easier to interpret.""",
                unsafe_allow_html = True)

    with Image.open('images/scatter_plots_vs_wage.png') as img:
        st.image(img, caption = """Scatter plots of critical features
                (x-axis) against Wage (y-axis)""")

    st.markdown("""Once important features were identified and strongly
                correlated features were pruned. I did a final correlation
                plot to better understand the relationship between those
                features and 'Wage'""")

    with Image.open('images/strong_correlation_heatmap.png') as img:
        st.image(img, caption = """Correlation of critical features
                (x-axis) against Wage (y-axis)""")

    st.markdown("""To prepare the data for analysis, I removed extraneous
                or non-informative features such as 'playerUrl'. The
                remaining set of features were then converted to meaninful
                numerical columns for use in statistical models. I scaled
                all meaningful features in case further exploration was
                desired. For example, 'A/W' and 'D/W' low, medium, and high
                values were converted to 0, 1, 2 respectively and I used
                OneHotEncoder to encode 'Best Position'. The remaining
                numerical features were encoded using MinMaxScaler.
                I then created the train/test split with sklearn and
                saved the training and test data for analysis.""")
