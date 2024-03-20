# YouTube Video Engagement Predictor

This is a Streamlit app that predicts YouTube video engagement based on video metadata. The app predicts the likes per viewer of a video using a machine learning model trained on data from the Kaggle competition "Predict YouTube Video Likes (Pog Series #1)", a competition supported by the Twitch data science community.

## Features

The app allows users to input the following video metadata:

- Video title
- Publishing datetime
- Channel name
- Video category
- Video tags
- Video duration
- Video description
- Prediction date

Based on these inputs, the app outputs a prediction for the likes per viewer of the video.

## Model

The machine learning model used in this app is LightGBM. It uses various features extracted from the inputs, including text features.

## Installation

To run this app, you need to install the required Python libraries. You can do this by running the following command:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Usage

To start the app, navigate to the app's directory and run the following command:

\`\`\`bash
streamlit run app.py
\`\`\`

Then, open your web browser and go to `http://localhost:8501` to access the app.

Alternatively, you can also use the app directly on the website at https://youtube-engagement-prediction.streamlit.app/
