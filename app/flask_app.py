from flask import Flask, jsonify, render_template, request

from student_performance_indicator.pipeline.prediction_pipeline import (
    CustomData,
    PredictPipeline,
)

app = Flask(__name__)


@app.route("/")
def home():
    """
    Render the home page of the web app.

    Returns:
        Rendered HTML template for the home page.
    """
    return render_template("home.html")


@app.route("/predict", methods=["GET"])
def predict_page():
    """
    Render the prediction input page where users can enter data.

    Returns:
        Rendered HTML template for the prediction form page.
    """
    return render_template("predict.html")


@app.route("/predictdata", methods=["POST"])
def predict_datapoint():
    """
    API endpoint to receive input data as JSON, process it through
    the prediction pipeline, and return the predicted math score as JSON.

    Expects a JSON payload with keys:
        - gender
        - race_ethnicity
        - parental_level_of_education
        - lunch
        - test_preparation_course
        - reading_score
        - writing_score

    Returns:
        JSON response containing the predicted math score or
        an error message with status code 500 in case of failure.
    """
    try:
        # Parse JSON input from the request
        data = request.get_json()

        # Create a CustomData instance from received data
        custom_data = CustomData(
            gender=data["gender"],
            race_ethnicity=data["race_ethnicity"],
            parental_level_of_education=data["parental_level_of_education"],
            lunch=data["lunch"],
            test_preparation_course=data["test_preparation_course"],
            reading_score=float(data["reading_score"]),
            writing_score=float(data["writing_score"]),
        )

        # Convert input data to DataFrame format expected by pipeline
        pred_df = custom_data.get_data_as_data_frame()

        # Initialize prediction pipeline and predict the target value
        pipeline = PredictPipeline()
        result = pipeline.predict(pred_df)

        # Clamp the prediction between 0 and 100
        result = max(0, min(100, result))

        # Return the prediction result as JSON
        return jsonify({"math_score": result[0]})

    except Exception as e:
        # Return error message as JSON with HTTP 500 status code
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Run the Flask app with debug mode enabled for development
    app.run(debug=True)
