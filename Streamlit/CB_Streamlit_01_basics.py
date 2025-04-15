import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def get_df(high_limit, low_limit, x_size, y_size):

    df = pd.DataFrame(
        np.random.randint(
            low = low_limit,
            high = high_limit,
            size = (x_size, y_size) 
        ),
        columns = ("col %d" % i for i in range (y_size))
    )

    return df

s = st.sidebar.chat_input(placeholder = "Your message")
if s:
    st.sidebar.success(s)

st.image("https://cdn.prod.website-files.com/63c2c7b1f3d9c51c32335fb0/664c5634a1dfc7c6526459ec_Conquer-Blocks-logo.avif")
st.title("Streamlit basics")
st.markdown("A brief project on Streamlit widgets :sunglasses:", unsafe_allow_html=False)

empty_widget = st.empty()

generate_df = st.checkbox("Generate df")

with st.expander("Data"):

    col1, col2, col3 = st.columns([2,6,2]) #cada 1 refleja 1 columna que vamos a tener

    with col1:

        low_limit = st.slider(
            "Low limit",
            min_value = -1000,
            max_value = 1000,
            value = -10,
            step = 1
        )

        high_limit = st.slider(
            "High limit",
            min_value = -1000,
            max_value = 1000,
            value = 10,
            step = 1
        )


        x_size = st.number_input(
            "X size",
            min_value = 1,
            max_value = 10**9,
            value = 10,
            step = 1
        )

        y_size = st.number_input(
            "Y size",
            min_value = 1,
            max_value = 10**9,
            value = 10,
            step = 1
        )



    with col2:
        with st.spinner("Generating dataframe..."):    
            if generate_df:
                df = get_df(high_limit, low_limit, x_size, y_size)
                st.dataframe(df)
                empty_widget.write("Dataframe dimensions %d x %d" % (df.shape))
                

    with col3:
        download_button = st.download_button(
            label = "Download df", 
            data = df.to_csv(),
            file_name = "My dataframe.csv"
        )

with st.expander("Metrics"):

    df_max = df.max().max()
    df_min = df.min().min()
    df_mean = df.mean().mean()

    metrics_selection = st.multiselect(
        label = "Select metrics to display",
        options = ["Max", "Min", "Mean"],
        placeholder= "Choose an option",
        )

    if 'Max' in metrics_selection:
        st.metric(
            label = "Max value",
            value = df_max,
            delta = df_max - df_mean,
            delta_color = "normal", 
            help = "The max value of the dataframe",
        )

    if 'Min' in metrics_selection:
        st.metric(
            label = "Min value",
            value = df_min,
            delta = df_min - df_mean,
            delta_color = "normal", 
            help = "The min value of the dataframe",
        )

    if 'Mean' in metrics_selection:
        st.metric(
            label = "Mean value",
            value = df_mean,
            delta = None,
            delta_color = "normal", 
            help = "The mean value of the dataframe",
        )

with st.expander("Plots"):


    colorscale = st.selectbox("Choose color", 
                            options = [
                                'viridis',
                                'cividis',
                                'inferno',
                                'magma',
                                'plasma',
                                'greys',
                                'blues',
                                'greens',
                                'oranges',
                                'reds',
                                'purples',
                                'rainbow',
                                'jet',                              
                            ])

    tab1, tab2, tab3 = st.tabs(["Matplotlib", "Plotly2D", "Plotly3D"])

    with tab1:
        
        fig = plt.figure()
        contour = plt.contour(
            df,
            cmap = colorscale
        )
        plt.colorbar(contour)
        st.pyplot(fig)

    with tab2:
        
        fig = go.Figure(
            data = 
            go.Contour(
                z = df,
                colorscale = colorscale
            )
        )
        st.plotly_chart(fig)
    with tab3:
    
        fig = go.Figure(
            data = 
            go.Surface(
                z = df,
                colorscale = colorscale,
            )
        )
        st.plotly_chart(fig)