# YourCabs.com Booking Cancellation Prediction

## Project Overview

This project aims to improve customer service at YourCabs.com, a cab company based in Bangalore, by predicting the likelihood of booking cancellations due to car unavailability. Booking cancellations can cause significant inconvenience to passengers, especially when they occur close to the trip start time. The predictive model built here helps the business classify new bookings and predict whether they will be canceled.

The data is categorized into three travel types:
1. Long-distance bookings
2. Point-to-point bookings
3. Hourly rental bookings

For each travel type, models like Decision Trees, Naive Bayes, and Random Forest are applied to identify the likelihood of cancellation.

## Dataset

The dataset provided contains various booking-related fields such as:
- `from_area_id`: The area ID from which the cab is booked.
- `to_area_id`: The destination area ID.
- `vehicle_model_id`: The ID of the vehicle model.
- `from_date`, `booking_created`: The booking creation date and the start date of the trip.
- `Car_Cancellation`: Binary label indicating whether the booking was canceled due to unavailability of a car.

Data pre-processing included removing irrelevant columns and creating features like:
- Cancellation percentages by area and route.
- Categories based on time of booking (urgent, same day, etc.).
- Temporal features like hour of day and day of the week.

## Feature Engineering

- **Area-wise cancellation percentage**: Calculated based on past cancellations for a given `from_area_id`.
- **Route-wise cancellation percentage**: Based on the combination of `from_area_id` and `to_area_id`.
- **Time features**: Created time-of-day, booking type (urgent, normal), day of the week, and weekend indicators.
- **Distance calculation**: For point-to-point bookings, the geodesic distance between the `from_lat`, `from_long` and `to_lat`, `to_long` was calculated.

## Models Used

The following machine learning models were trained to predict cancellations:
- **Decision Tree Classifier**
- **Naive Bayes Classifier**
- **Random Forest Classifier**

Each model was trained on different subsets of data (long-distance, point-to-point, hourly rental) after relevant feature extraction and pre-processing.

### Model Evaluation
Model performance was evaluated using standard classification metrics such as:
- **Precision**
- **Recall**
- **F1 Score**
- **Accuracy**

## Results

| Model                  | Travel Type         | Accuracy | Precision | Recall | F1-Score |
|------------------------|---------------------|----------|-----------|--------|----------|
| Decision Tree           | Long Distance       | 98%      | 50%       | 50%    | 50%      |
| Random Forest           | Point-to-Point      | 96%      | 90%       | 80%    | 84%      |
| Random Forest           | Hourly Rental       | 95%      | 68%       | 57%    | 59%      |
