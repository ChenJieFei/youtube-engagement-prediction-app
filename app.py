import model
import streamlit as st
import pandas as pd
from datetime import datetime

def app():
    st.markdown("# Youtube Engagement:")
    st.markdown("## Likes Per View Predictor")

    with st.form(key='my_form'):
        st.header("Video Details")
        title = st.text_input(label='Video Title')
        publishedDate = st.date_input('Publishing Date')
        channelTitle = st.text_input(label='Channel Name')
        publishedTime = st.time_input('Publishing Time')
        publishedAt = datetime.combine(publishedDate, publishedTime)

        categories = {
            "Film & Animation": 1,
            "Autos & Vehicles": 2,
            "Music": 10,
            "Pets & Animals": 17,
            "Sports": 19,
            "Travel & Events": 20,
            "Gaming": 22,
            "People & Blogs": 23,
            "Comedy": 24,
            "Entertainment": 25,
            "News & Politics": 26,
            "How to & Style": 27,
            "Education": 28,
            "Science & Technology": 3,
            "Nonprofits & Activism": 21
        }

        category = st.selectbox('Video Category', list(categories.keys()))
        categoryId = categories[category]

        tags = st.text_input(label='Enter Tags (separated by commas)')
        tags = tags.replace(' ', '')
        tags = tags.replace(',','|')
        duration_seconds = int(st.number_input(label='Video Duration (seconds)', format="%i", step=1))
        description = st.text_area(label='Video Description')
        trending_date = st.date_input(label='Prediction Date')
        submit_button = st.form_submit_button(label='Calculate')

        if submit_button:
            if not title or not channelTitle:
                st.error('Video title and channel name cannot be empty.')
            elif publishedAt.date() > trending_date:
                st.error('Prediction date must be on or after the publishing date.')
            else:
                data = {
                    'title': title,
                    'publishedAt': pd.to_datetime(publishedAt, utc=True),
                    'channelTitle': channelTitle,
                    'categoryId': categoryId,
                    'trending_date': pd.to_datetime(trending_date).date(),
                    'tags': [tags],
                    'duration_seconds': duration_seconds,
                    'description': description,
                    'ratings_disabled': False,
                    'comments_disabled': False,
                    'has_thumbnail': True
                }
                df = pd.DataFrame(data, index=[0])

                prediction = model.main(df)
                st.subheader("Prediction")
                st.balloons()
                st.success(f'Predicted Likes per View: {round(prediction[0], 4)}')

if __name__ == "__main__":
    app()
