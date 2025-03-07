6. Open your browser and navigate to `http://127.0.0.1:5000/`

## Dataset

The application uses a sample dataset located in `data/salary_data.csv` that contains the following features:
- Years of experience
- Education level (Bachelors, Masters, PhD)
- Job role (Software Developer, Data Scientist)
- City tier (1 for metro cities, 2 for smaller cities)
- Salary (target variable)

## Model

The application uses a Linear Regression model with preprocessing of categorical features using OneHotEncoder.

## Deployment

This application can be deployed to any platform that supports Python and Flask, such as Heroku, AWS, or Google Cloud Platform.
