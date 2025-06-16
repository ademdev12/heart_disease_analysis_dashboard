# Heart Disease Analysis Dashboard

This project provides a comprehensive dashboard for analyzing heart disease data and predicting heart disease risk using a Streamlit application 

## Features

- **Interactive Dashboard**: Built with Streamlit, allowing users to explore heart disease data through various filters and visualizations.
- **Key Performance Indicators (KPIs)**: Displays prevalence, average age, average cholesterol, and patient count.
- **Detailed Analysis**: Provides histograms, box plots, violin plots, scatter plots, bar plots, pie charts, and heatmaps for in-depth data exploration.
- **Heart Disease Prediction**: A machine learning model (Random Forest Classifier) to predict heart disease risk based on patient input.

## Project Structure

```
heart_disease_analysis_dashboard/
├── app.py
├── data/
│   └── heart_disease_uci.csv
└── requirements.txt
```

- `app.py`: The main Streamlit application file.
- `data/heart_disease_uci.csv`: The dataset used for analysis and prediction.
- `requirements.txt`: Lists all the Python dependencies required to run the project.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/heart_disease_analysis_dashboard.git
   cd heart_disease_analysis_dashboard
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Streamlit Dashboard

To run the interactive dashboard, execute the following command in your terminal:

```bash
streamlit run app.py
```

This will open the dashboard in your web browser.

## Dataset

The `heart_disease_uci.csv` dataset contains various attributes related to heart disease, including:

- `age`: Age of the patient.
- `sex`: Sex of the patient (Male/Female).
- `cp`: Chest pain type.
- `trestbps`: Resting blood pressure.
- `chol`: Serum cholesterol.
- `fbs`: Fasting blood sugar.
- `restecg`: Resting electrocardiographic results.
- `thalch`: Maximum heart rate achieved.
- `exang`: Exercise induced angina.
- `oldpeak`: ST depression induced by exercise relative to rest.
- `slope`: Slope of the peak exercise ST segment.
- `ca`: Number of major vessels colored by fluoroscopy.
- `thal`: Thallium stress test result.
- `num`: Heart disease diagnosis (0 for absence, 1-4 for presence).

## Machine Learning Model

The project uses a Random Forest Classifier for heart disease prediction. The model is trained on the provided dataset and integrated into the Streamlit dashboard for real-time predictions.

## Contributing

Feel free to fork this repository, submit pull requests, or open issues for any improvements or bug fixes.

## License

Distributed under the MIT License. See LICENSE for more information.


