# Streamlit Sign Language Recognition Application

This project is a Streamlit application for real-time sign language recognition using a pre-trained model. The application captures video input from a webcam, processes the frames to extract hand features, and predicts the corresponding sign language gestures.

## Project Structure

```
streamlit-slr-app
├── src
│   ├── app.py          # Main entry point for the Streamlit application
│   ├── deploy_model.py  # Contains the sign language recognition model code
│   └── utils.py        # Utility functions for data processing and visualization
├── requirements.txt    # Python dependencies for the project
└── README.md           # Documentation for the project
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd streamlit-slr-app
   ```

2. **Create a virtual environment (optional but recommended):**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit application:**
   ```
   streamlit run src/app.py
   ```

2. **Access the application:**
   Open your web browser and go to `http://localhost:8501` to interact with the sign language recognition application.

## Features

- Real-time video capture from the webcam.
- Hand detection and feature extraction using MediaPipe.
- Sign language gesture recognition using a pre-trained TensorFlow model.
- User-friendly interface with Streamlit for displaying predictions and progress.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.