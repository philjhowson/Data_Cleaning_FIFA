import streamlit as st
from PIL import Image

def comparison():

    st.markdown("""<h2 style = "font-size: 1.5em;"><u>
                Camprison of Statistical Models</u></h2>""",
                unsafe_allow_html = True)

    st.markdown("""I performed a statistical analysis using three models:<br>
                1. Random Forests Regressor<br>
                2. XGBoost Regressor<br>
                3. FNN<br>
                For the Random Forests Regressor and XGBoost Regressor, I
                performed a grid search to better tune the hyperparameters
                and find the best model. However, in the end, each model
                performed similarily. Although, the first
                two models did suffer slightly from overfitting, which can
                be observed by the fact that the training scores were higher
                than the test scores. The FNN, on the other hand, had a
                higher test score than training score, indicating good
                generalization of the learned features. The figure below
                presents the training and test scores for each model.""",
                unsafe_allow_html = True)

    with Image.open('images/training_test_scores.png') as img:
        st.image(img, caption = """Training and test scores for each of
                 the three models I generated""")

    st.markdown("""I also extracted the feature importances for each of the
                models. I extracted the feature importances for the first two
                models by checking the .feature_importances_ attribute, but
                for the FNN, I used the shap library. In general, there was
                agreement across the models as to the most important features,
                but the FNN but more weight on the feature 'POT', and the
                Random Forests Regressor put more weight on the feature 'Value'.
                However, the overall relative importance was similar across
                models. The figures below present the feature importances for
                each model and the shap values for the FNN model.""")

    with Image.open('images/feature_importance.png') as img:
        st.image(img, caption = """The relative feature importances for
                 each model.""")

    with Image.open('images/shap_summary_plot.png') as img:
        st.image(img, caption = """FNN Shap Values for each feature.""")

    st.markdown("""<h2 style = "font-size: 1.5em;"><u>
                Conclusions</u></h2>""",
                unsafe_allow_html = True)

    st.markdown("""Overall, the statistical analyses strongly suggests
                that key indicators of salary in FIFA are directly related
                to a player's skill level. This is clear because the strongest
                feature in all the models was 'OVA.' Another strong indicator
                of salary, was 'Value.' The feature represents how much
                value (in euros) the player brings in to the club. It therefore
                makes a great deal of sense that the more value a player
                brings to the club, the higher their compensation would be.<br>
                For future work, more exploration of the feature space and
                possible interactions between other features not tested here
                could further explain what factors contribute to player
                salaries.""", unsafe_allow_html = True)
