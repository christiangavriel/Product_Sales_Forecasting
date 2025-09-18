# import streamlit as st
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from io import StringIO
# import datetime

# # Set page config
# st.set_page_config(
#     page_title="Electronic Sales Analysis",
#     page_icon="📊",
#     layout="wide"
# )

# # Page title
# st.title("🛒 Electronic Sales Data Analysis & Forecasting")
# st.markdown("Upload your electronic sales data to analyze trends and forecast future sales.")

# # File uploader
# uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

# if uploaded_file is not None:
#     # Read data
#     try:
#         all_data = pd.read_csv(uploaded_file)
        
#         # Show raw data
#         st.subheader("Raw Data Preview")
#         st.dataframe(all_data.head())
        
#         # Data cleaning tab
#         tab1, tab2, tab3, tab4 = st.tabs(["Data Cleaning", "Sales Analysis", "Product Analysis", "Sales Forecasting"])
        
#         with tab1:
#             st.header("Data Cleaning and Preprocessing")
            
#             # Drop columns that are irrelevant
#             col1, col2 = st.columns(2)
#             with col1:
#                 columns_to_drop = st.multiselect(
#                     "Select columns to drop",
#                     options=all_data.columns.tolist(),
#                     default=["Customer ID", "Shipping Type"] if "Customer ID" in all_data.columns and "Shipping Type" in all_data.columns else []
#                 )
            
#             if columns_to_drop:
#                 all_data = all_data.drop(columns_to_drop, axis=1)
#                 st.success(f"Dropped columns: {', '.join(columns_to_drop)}")
            
#             # Convert date column if it exists
#             with col2:
#                 date_columns = st.multiselect(
#                     "Select date columns to convert",
#                     options=all_data.columns.tolist(),
#                     default=["Purchase Date"] if "Purchase Date" in all_data.columns else []
#                 )
            
#             for date_col in date_columns:
#                 try:
#                     all_data[date_col] = pd.to_datetime(all_data[date_col])
#                     st.success(f"Converted {date_col} to datetime.")
#                 except:
#                     st.error(f"Could not convert {date_col} to datetime.")
            
#             # Check for missing values
#             st.subheader("Missing Values")
#             missing_values = all_data.isnull().sum()
#             st.write(missing_values)
            
#             if missing_values.sum() > 0:
#                 handle_missing = st.radio(
#                     "How to handle missing values?",
#                     options=["Drop rows with missing values", "Keep as is"]
#                 )
                
#                 if handle_missing == "Drop rows with missing values":
#                     original_rows = len(all_data)
#                     all_data.dropna(inplace=True)
#                     st.success(f"Dropped {original_rows - len(all_data)} rows with missing values.")
            
#             # Check for duplicates
#             st.subheader("Duplicate Rows")
#             duplicate_count = all_data.duplicated().sum()
#             st.write(f"Number of duplicate rows: {duplicate_count}")
            
#             if duplicate_count > 0:
#                 handle_duplicates = st.radio(
#                     "How to handle duplicate rows?",
#                     options=["Drop duplicate rows", "Keep duplicates"]
#                 )
                
#                 if handle_duplicates == "Drop duplicate rows":
#                     original_rows = len(all_data)
#                     all_data = all_data.drop_duplicates()
#                     st.success(f"Dropped {original_rows - len(all_data)} duplicate rows.")
            
#             # Feature Engineering
#             st.subheader("Feature Engineering")
            
#             if "Purchase Date" in all_data.columns and pd.api.types.is_datetime64_any_dtype(all_data["Purchase Date"]):
#                 add_features = st.checkbox("Add date-based features", value=True)
                
#                 if add_features:
#                     all_data['Months'] = all_data['Purchase Date'].dt.month_name()
#                     all_data['Day of Week'] = all_data['Purchase Date'].dt.day_name()
#                     all_data['Week'] = all_data['Purchase Date'].dt.isocalendar().week
#                     all_data['Day'] = all_data['Purchase Date'].dt.day
                    
#                     st.success("Added date features: Months, Day of Week, Week, Day")
            
#             # Show cleaned data
#             st.subheader("Cleaned Data Preview")
#             st.dataframe(all_data.head())
            
#             # Download cleaned data
#             csv = all_data.to_csv(index=False)
#             st.download_button(
#                 label="Download cleaned data as CSV",
#                 data=csv,
#                 file_name="cleaned_data.csv",
#                 mime="text/csv"
#             )
        
#         with tab2:
#             st.header("Sales Analysis")
            
#             if "Purchase Date" in all_data.columns and "Total Price" in all_data.columns:
#                 # Daily sales analysis
#                 st.subheader("Daily Sales Trend")
                
#                 if 'Day' in all_data.columns:
#                     daily_sales = all_data.groupby('Day')['Total Price'].sum()
#                     daily_sales = daily_sales[daily_sales.index <= 30]
#                     daily_sales_smooth = daily_sales.rolling(window=3, min_periods=1).mean()
                    
#                     fig1, ax1 = plt.subplots(figsize=(10, 6))
#                     ax1.plot(daily_sales.index, daily_sales_smooth, marker='o', linestyle='-')
#                     ax1.set_xlabel("Day", fontsize=11)
#                     ax1.set_ylabel("Total Sales", fontsize=11)
#                     ax1.set_title("Total Daily Sales per Day", fontweight='bold')
#                     ax1.grid(True)
#                     st.pyplot(fig1)
                
#                 # Weekly sales analysis
#                 st.subheader("Weekly Sales Distribution")
                
#                 if 'Week' in all_data.columns:
#                     weekly_sales = all_data.groupby('Week').agg({'Total Price': 'sum'}).reset_index()
                    
#                     fig2, ax2 = plt.subplots(figsize=(10, 6))
#                     sns.barplot(x='Week', y='Total Price', data=weekly_sales, estimator=np.sum, ax=ax2)
#                     ax2.set_xlabel("Week of the Year", fontsize=11)
#                     ax2.set_ylabel("Total Sales", fontsize=11)
#                     ax2.set_title("Weekly Sales Distribution for the Year", fontweight='bold')
#                     plt.xticks(rotation=45)
#                     st.pyplot(fig2)
                
#                 # Day of week analysis
#                 st.subheader("Sales by Day of Week")
                
#                 if 'Day of Week' in all_data.columns:
#                     fig3, ax3 = plt.subplots(figsize=(10, 6))
#                     sns.barplot(x='Day of Week', y='Total Price', data=all_data, estimator=np.sum, ax=ax3)
#                     ax3.set_xlabel("Day of Week", fontsize=11)
#                     ax3.set_ylabel("Total Sales", fontsize=11)
#                     ax3.set_title("Sales Distributed Based on Day of The Week", fontweight='bold')
#                     plt.xticks(rotation=45)
#                     st.pyplot(fig3)
        
#         with tab3:
#             st.header("Product Analysis")
            
#             if "Product Type" in all_data.columns:
#                 # Best selling products
#                 st.subheader("Best Selling Products by Quantity")
                
#                 if "Quantity" in all_data.columns:
#                     product_quantity = all_data.groupby('Product Type')['Quantity'].sum().sort_values(ascending=False)
                    
#                     fig4, ax4 = plt.subplots(figsize=(10, 6))
#                     sns.barplot(x='Product Type', y='Quantity', data=all_data, estimator=sum, ax=ax4)
#                     ax4.set_xlabel('Product Type', fontsize=11)
#                     ax4.set_ylabel('Total Quantity Sold', fontsize=11)
#                     ax4.set_title('Total Quantity Sold per Product Type', fontweight='bold')
#                     plt.xticks(rotation=45)
#                     st.pyplot(fig4)
                    
#                     st.write("Top selling products by quantity:")
#                     st.dataframe(product_quantity.reset_index().rename(columns={'Product Type': 'Product', 'Quantity': 'Total Quantity'}))
                
#                 # Product revenue
#                 st.subheader("Product Revenue Analysis")
                
#                 if "Total Price" in all_data.columns:
#                     product_revenue = all_data.groupby('Product Type')['Total Price'].sum().sort_values(ascending=False)
                    
#                     fig5, ax5 = plt.subplots(figsize=(10, 6))
#                     sns.barplot(x='Product Type', y='Total Price', data=all_data, estimator=sum, ax=ax5)
#                     ax5.set_xlabel('Product Type', fontsize=11)
#                     ax5.set_ylabel('Total Revenue', fontsize=11)
#                     ax5.set_title('Total Revenue by Product Type', fontweight='bold')
#                     plt.xticks(rotation=45)
#                     st.pyplot(fig5)
                    
#                     st.write("Products by revenue:")
#                     st.dataframe(product_revenue.reset_index().rename(columns={'Product Type': 'Product', 'Total Price': 'Total Revenue'}))
                
#                 # Weekly product trends
#                 if 'Week' in all_data.columns and "Quantity" in all_data.columns:
#                     st.subheader("Weekly Product Trends")
                    
#                     weekly_best_selling = all_data.groupby(['Week', 'Product Type'])['Quantity'].sum().reset_index()
                    
#                     fig6, ax6 = plt.subplots(figsize=(10, 6))
#                     sns.lineplot(data=weekly_best_selling, x='Week', y='Quantity', hue='Product Type', marker='o', ax=ax6)
#                     ax6.set_xlabel('Week', fontsize=11)
#                     ax6.set_ylabel('Total Quantity Sold', fontsize=11)
#                     ax6.set_title('Weekly Best Selling Products', fontweight='bold')
#                     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#                     plt.grid(True)
#                     st.pyplot(fig6)
        
#         with tab4:
#             st.header("Sales Forecasting")
            
#             if "Product Type" in all_data.columns and "Purchase Date" in all_data.columns and "Quantity" in all_data.columns:
#                 st.info("This tab uses machine learning to forecast future product sales based on historical patterns.")
                
#                 # Check if we have enough data for forecasting
#                 if len(all_data) < 20:
#                     st.warning("Not enough data for accurate forecasting. Please upload a dataset with more records.")
#                 else:
#                     # Prepare data for ML model
#                     le = LabelEncoder()
#                     all_data["Product Type Encoded"] = le.fit_transform(all_data["Product Type"])
                    
#                     # Group by date and product
#                     st.write("Creating time series features...")
#                     daily_sales = all_data.groupby(["Purchase Date", "Product Type Encoded"]).agg({"Quantity": "sum"}).reset_index()
                    
#                     # Add date features if they don't exist
#                     if 'Months' not in daily_sales.columns:
#                         daily_sales['Months'] = daily_sales['Purchase Date'].dt.month_name()
#                     if 'Day of Week' not in daily_sales.columns:
#                         daily_sales['Day of Week'] = daily_sales['Purchase Date'].dt.day_name()
#                     if 'Week' not in daily_sales.columns:
#                         daily_sales['Week'] = daily_sales['Purchase Date'].dt.isocalendar().week
#                     if 'Day' not in daily_sales.columns:
#                         daily_sales['Day'] = daily_sales['Purchase Date'].dt.day
                    
#                     # Create lag features
#                     for lag in range(1, 8):
#                         daily_sales[f'lag_{lag}'] = daily_sales.groupby("Product Type Encoded")["Quantity"].shift(lag)
                    
#                     # Create moving average
#                     daily_sales['moving_avg_7'] = daily_sales.groupby("Product Type Encoded")["Quantity"].transform(lambda x: x.rolling(7).mean())
                    
#                     # Drop rows with NaN (from lag creation)
#                     daily_sales.dropna(inplace=True)
                    
#                     # One-hot encode categorical features
#                     daily_sales = pd.get_dummies(daily_sales, columns=['Months', 'Day of Week'], drop_first=True)
                    
#                     # Define features and target
#                     features = [col for col in daily_sales.columns if col not in ["Purchase Date", "Quantity", "Product Type Encoded"]]
#                     target = "Quantity"
                    
#                     X = daily_sales[features]
#                     y = daily_sales[target]
                    
#                     # Train-test split
#                     st.write("Splitting data into training and testing sets...")
#                     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
                    
#                     # Train model
#                     with st.spinner("Training the XGBoost model..."):
#                         model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
#                         model.fit(X_train, y_train)
                    
#                     # Make predictions on test set
#                     test_predictions = model.predict(X_test)
                    
#                     # Evaluate model
#                     mae = mean_absolute_error(y_test, test_predictions)
#                     rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
#                     r2 = r2_score(y_test, test_predictions)
                    
#                     # Display metrics
#                     st.subheader("Model Performance")
#                     col1, col2, col3 = st.columns(3)
#                     col1.metric("Mean Absolute Error", f"{mae:.2f}")
#                     col2.metric("Root Mean Squared Error", f"{rmse:.2f}")
#                     col3.metric("R² Score", f"{r2:.2f}")
                    
#                     # Make future predictions
#                     st.subheader("Sales Forecast")
                    
#                     # Select number of days to forecast
#                     forecast_days = st.slider("Number of days to forecast", min_value=1, max_value=14, value=7)
                    
#                     future_dates = pd.date_range(start=daily_sales["Purchase Date"].max(), periods=forecast_days+1, freq='D')[1:]
#                     future_predictions = []
                    
#                     with st.spinner("Generating forecasts..."):
#                         for product_type in daily_sales["Product Type Encoded"].unique():
#                             latest_data = daily_sales[daily_sales["Product Type Encoded"] == product_type].iloc[-1:].copy()
                            
#                             forecast_features = latest_data[features].copy()
                            
#                             for date in future_dates:
#                                 prediction = model.predict(forecast_features)[0]
                                
#                                 future_predictions.append({
#                                     "Purchase Date": date,
#                                     "Product Type": le.inverse_transform([product_type])[0],
#                                     "Predicted Quantity": prediction
#                                 })
                                
#                                 new_features = forecast_features.copy()
#                                 for lag in range(7, 1, -1):
#                                     new_features[f'lag_{lag}'] = new_features[f'lag_{lag-1}']
#                                 new_features['lag_1'] = prediction
#                                 new_features['moving_avg_7'] = new_features[[f'lag_{i}' for i in range(1, 8)]].mean().values[0]
                                
#                                 forecast_features = new_features
                    
#                     forecast_df = pd.DataFrame(future_predictions)
                    
#                     st.write("Forecast for the next few days:")
#                     st.dataframe(forecast_df)
                    
#                     # Visualize forecast
#                     st.subheader("Forecast Visualization")
                    
#                     products_to_plot = st.multiselect(
#                         "Select products to visualize",
#                         options=forecast_df["Product Type"].unique(),
#                         default=forecast_df["Product Type"].unique()[:min(3, len(forecast_df["Product Type"].unique()))]
#                     )
                    
#                     if products_to_plot:
#                         fig, ax = plt.subplots(figsize=(10, 6))
                        
#                         for product in products_to_plot:
#                             product_forecast = forecast_df[forecast_df["Product Type"] == product]
#                             ax.plot(product_forecast["Purchase Date"], product_forecast["Predicted Quantity"], 
#                                    marker='o', label=product)
                        
#                         ax.set_xlabel("Date")
#                         ax.set_ylabel("Predicted Quantity")
#                         ax.set_title("Sales Forecast by Product Type")
#                         ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#                         ax.grid(True)
#                         plt.xticks(rotation=45)
#                         plt.tight_layout()
#                         st.pyplot(fig)
                        
#                         # Download forecast data
#                         csv = forecast_df.to_csv(index=False)
#                         st.download_button(
#                             label="Download forecast data as CSV",
#                             data=csv,
#                             file_name="sales_forecast.csv",
#                             mime="text/csv"
#                         )
    
#     except Exception as e:
#         st.error(f"Error processing the file: {str(e)}")
#         st.error("Make sure your CSV file has the expected columns: Purchase Date, Product Type, Quantity, Total Price")

# else:
#     st.info("Please upload a CSV file to begin analysis. The file should contain electronic sales data with columns such as Purchase Date, Product Type, Quantity, and Total Price.")
    
#     # Sample data option
#     st.write("Need sample data? You can use these columns as a reference:")
#     sample_df = pd.DataFrame({
#         'Purchase Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
#         'Product Type': ['Smartphone', 'Laptop', 'Headphones'],
#         'Quantity': [2, 1, 3],
#         'Total Price': [1200.00, 1500.00, 150.00]
#     })
#     st.dataframe(sample_df)

# # Footer
# st.markdown("---")
# st.markdown("📊 Electronic Sales Analysis & Forecasting App | Created with Streamlit")

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from io import StringIO
import datetime

# Set page config
st.set_page_config(
    page_title="Electronic Sales Analysis",
    page_icon="📊",
    layout="wide"
)

# Page title
st.title("🛒 Electronic Sales Data Analysis & Forecasting")
st.markdown("Upload your electronic sales data to analyze trends and forecast future sales.")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    # Read data
    try:
        all_data = pd.read_csv(uploaded_file)
        
        # Show raw data
        st.subheader("Raw Data Preview")
        st.dataframe(all_data.head())
        
        # Data cleaning tab
        tab1, tab2, tab3, tab4 = st.tabs(["Data Cleaning", "Sales Analysis", "Product Analysis", "Sales Forecasting"])
        
        with tab1:
            st.header("Data Cleaning and Preprocessing")
            
            # Drop columns that are irrelevant
            col1, col2 = st.columns(2)
            with col1:
                columns_to_drop = st.multiselect(
                    "Select columns to drop",
                    options=all_data.columns.tolist(),
                    default=["Customer ID", "Shipping Type"] if "Customer ID" in all_data.columns and "Shipping Type" in all_data.columns else []
                )
            
            if columns_to_drop:
                all_data = all_data.drop(columns_to_drop, axis=1)
                st.success(f"Dropped columns: {', '.join(columns_to_drop)}")
            
            # Convert date column if it exists
            with col2:
                date_columns = st.multiselect(
                    "Select date columns to convert",
                    options=all_data.columns.tolist(),
                    default=["Purchase Date"] if "Purchase Date" in all_data.columns else []
                )
            
            for date_col in date_columns:
                try:
                    all_data[date_col] = pd.to_datetime(all_data[date_col])
                    st.success(f"Converted {date_col} to datetime.")
                except:
                    st.error(f"Could not convert {date_col} to datetime.")
            
            # Check for missing values
            st.subheader("Missing Values")
            missing_values = all_data.isnull().sum()
            st.write(missing_values)
            
            if missing_values.sum() > 0:
                handle_missing = st.radio(
                    "How to handle missing values?",
                    options=["Drop rows with missing values", "Keep as is"]
                )
                
                if handle_missing == "Drop rows with missing values":
                    original_rows = len(all_data)
                    all_data.dropna(inplace=True)
                    st.success(f"Dropped {original_rows - len(all_data)} rows with missing values.")
            
            # Check for duplicates
            st.subheader("Duplicate Rows")
            duplicate_count = all_data.duplicated().sum()
            st.write(f"Number of duplicate rows: {duplicate_count}")
            
            if duplicate_count > 0:
                handle_duplicates = st.radio(
                    "How to handle duplicate rows?",
                    options=["Drop duplicate rows", "Keep duplicates"]
                )
                
                if handle_duplicates == "Drop duplicate rows":
                    original_rows = len(all_data)
                    all_data = all_data.drop_duplicates()
                    st.success(f"Dropped {original_rows - len(all_data)} duplicate rows.")
            
            # Feature Engineering
            st.subheader("Feature Engineering")
            
            if "Purchase Date" in all_data.columns and pd.api.types.is_datetime64_any_dtype(all_data["Purchase Date"]):
                add_features = st.checkbox("Add date-based features", value=True)
                
                if add_features:
                    all_data['Months'] = all_data['Purchase Date'].dt.month_name()
                    all_data['Day of Week'] = all_data['Purchase Date'].dt.day_name()
                    all_data['Week'] = all_data['Purchase Date'].dt.isocalendar().week
                    all_data['Day'] = all_data['Purchase Date'].dt.day
                    
                    st.success("Added date features: Months, Day of Week, Week, Day")
            
            # Show cleaned data
            st.subheader("Cleaned Data Preview")
            st.dataframe(all_data.head())
            
            # Download cleaned data
            csv = all_data.to_csv(index=False)
            st.download_button(
                label="Download cleaned data as CSV",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )
        
        with tab2:
            st.header("Sales Analysis")
            
            if "Purchase Date" in all_data.columns and "Total Price" in all_data.columns:
                # Daily sales analysis
                st.subheader("Daily Sales Trend")
                
                if 'Day' in all_data.columns:
                    daily_sales = all_data.groupby('Day')['Total Price'].sum()
                    daily_sales = daily_sales[daily_sales.index <= 30]
                    daily_sales_smooth = daily_sales.rolling(window=3, min_periods=1).mean()
                    
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    ax1.plot(daily_sales.index, daily_sales_smooth, marker='o', linestyle='-')
                    ax1.set_xlabel("Day", fontsize=11)
                    ax1.set_ylabel("Total Sales", fontsize=11)
                    ax1.set_title("Total Daily Sales per Day", fontweight='bold')
                    ax1.grid(True)
                    st.pyplot(fig1)
                
                # Weekly sales analysis
                st.subheader("Weekly Sales Distribution")
                
                if 'Week' in all_data.columns:
                    weekly_sales = all_data.groupby('Week').agg({'Total Price': 'sum'}).reset_index()
                    
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Week', y='Total Price', data=weekly_sales, estimator=np.sum, ax=ax2)
                    ax2.set_xlabel("Week of the Year", fontsize=11)
                    ax2.set_ylabel("Total Sales", fontsize=11)
                    ax2.set_title("Weekly Sales Distribution for the Year", fontweight='bold')
                    plt.xticks(rotation=45)
                    st.pyplot(fig2)
                
                # Day of week analysis
                st.subheader("Sales by Day of Week")
                
                if 'Day of Week' in all_data.columns:
                    fig3, ax3 = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Day of Week', y='Total Price', data=all_data, estimator=np.sum, ax=ax3)
                    ax3.set_xlabel("Day of Week", fontsize=11)
                    ax3.set_ylabel("Total Sales", fontsize=11)
                    ax3.set_title("Sales Distributed Based on Day of The Week", fontweight='bold')
                    plt.xticks(rotation=45)
                    st.pyplot(fig3)
        
        with tab3:
            st.header("Product Analysis")
            
            if "Product Type" in all_data.columns:
                # Best selling products
                st.subheader("Best Selling Products by Quantity")
                
                if "Quantity" in all_data.columns:
                    product_quantity = all_data.groupby('Product Type')['Quantity'].sum().sort_values(ascending=False)
                    
                    fig4, ax4 = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Product Type', y='Quantity', data=all_data, estimator=sum, ax=ax4)
                    ax4.set_xlabel('Product Type', fontsize=11)
                    ax4.set_ylabel('Total Quantity Sold', fontsize=11)
                    ax4.set_title('Total Quantity Sold per Product Type', fontweight='bold')
                    plt.xticks(rotation=45)
                    st.pyplot(fig4)
                    
                    st.write("Top selling products by quantity:")
                    st.dataframe(product_quantity.reset_index().rename(columns={'Product Type': 'Product', 'Quantity': 'Total Quantity'}))
                
                
                
                # Weekly product trends
                if 'Week' in all_data.columns and "Quantity" in all_data.columns:
                    st.subheader("Weekly Product Trends")
                    
                    weekly_best_selling = all_data.groupby(['Week', 'Product Type'])['Quantity'].sum().reset_index()
                    
                    fig6, ax6 = plt.subplots(figsize=(10, 6))
                    sns.lineplot(data=weekly_best_selling, x='Week', y='Quantity', hue='Product Type', marker='o', ax=ax6)
                    ax6.set_xlabel('Week', fontsize=11)
                    ax6.set_ylabel('Total Quantity Sold', fontsize=11)
                    ax6.set_title('Weekly Best Selling Products', fontweight='bold')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.grid(True)
                    st.pyplot(fig6)

                if "Product Type" in all_data.columns:
                    # Product revenue
                    if "Total Price" in all_data.columns:
                        st.subheader("Product Revenue")
                        product_revenue = all_data.groupby('Product Type')['Total Price'].sum().sort_values(ascending=False)
                        
                        fig5, ax5 = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='Product Type', y='Total Price', data=all_data, estimator=sum, ax=ax5)
                        ax5.set_xlabel('Product Type', fontsize=11)
                        ax5.set_ylabel('Total Revenue', fontsize=11)
                        ax5.set_title('Total Revenue by Product Type', fontweight='bold')
                        plt.xticks(rotation=45)
                        st.pyplot(fig5)
                        
                        st.write("Products by revenue:")
                        st.dataframe(product_revenue.reset_index().rename(columns={'Product Type': 'Product', 'Total Price': 'Total Revenue'}))

                    if 'Week' in all_data.columns and "Total Price" in all_data.columns:
                        st.subheader("Weekly Revenue per Product")
                        weekly_revenue = all_data.groupby(['Week', 'Product Type'])['Total Price'].sum().reset_index()

                        fig6, ax6 = plt.subplots(figsize=(10, 6))
                        sns.lineplot(data=weekly_revenue, x='Week', y='Total Price', hue='Product Type', marker='o', ax=ax6)
                        ax6.set_xlabel('Week', fontsize=11)
                        ax6.set_ylabel('Total Revenue', fontsize=11)
                        ax6.set_title('Weekly Revenue per Product', fontweight='bold')
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        plt.grid(True)
                        st.pyplot(fig6)
            
        
        with tab4:
            st.header("Sales Forecasting")
            
            if "Product Type" in all_data.columns and "Purchase Date" in all_data.columns and "Quantity" in all_data.columns:
                st.info("This tab uses machine learning to forecast future product sales based on historical patterns.")
                
                # Check if we have enough data for forecasting
                if len(all_data) < 20:
                    st.warning("Not enough data for accurate forecasting. Please upload a dataset with more records.")
                else:
                    # Prepare data for ML model
                    le = LabelEncoder()
                    all_data["Product Type Encoded"] = le.fit_transform(all_data["Product Type"])
                    
                    # Group by date and product
                    st.write("Creating time series features...")
                    daily_sales = all_data.groupby(["Purchase Date", "Product Type Encoded"]).agg({"Quantity": "sum"}).reset_index()
                    
                    # Add date features if they don't exist
                    if 'Months' not in daily_sales.columns:
                        daily_sales['Months'] = daily_sales['Purchase Date'].dt.month_name()
                    if 'Day of Week' not in daily_sales.columns:
                        daily_sales['Day of Week'] = daily_sales['Purchase Date'].dt.day_name()
                    if 'Week' not in daily_sales.columns:
                        daily_sales['Week'] = daily_sales['Purchase Date'].dt.isocalendar().week
                    if 'Day' not in daily_sales.columns:
                        daily_sales['Day'] = daily_sales['Purchase Date'].dt.day
                    
                    # Create lag features
                    for lag in range(1, 8):
                        daily_sales[f'lag_{lag}'] = daily_sales.groupby("Product Type Encoded")["Quantity"].shift(lag)
                    
                    # Create moving average
                    daily_sales['moving_avg_7'] = daily_sales.groupby("Product Type Encoded")["Quantity"].transform(lambda x: x.rolling(7).mean())
                    
                    # Drop rows with NaN (from lag creation)
                    daily_sales.dropna(inplace=True)
                    
                    # One-hot encode categorical features
                    daily_sales = pd.get_dummies(daily_sales, columns=['Months', 'Day of Week'], drop_first=True)
                    
                    # Define features and target
                    features = [col for col in daily_sales.columns if col not in ["Purchase Date", "Quantity", "Product Type Encoded"]]
                    target = "Quantity"
                    
                    X = daily_sales[features]
                    y = daily_sales[target]
                    
                    # Train-test split
                    st.write("Splitting data into training and testing sets...")
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
                    
                    # Train model
                    with st.spinner("Training the XGBoost model..."):
                        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
                        model.fit(X_train, y_train)
                    
                    # Make predictions on test set
                    test_predictions = model.predict(X_test)
                    
                    # Evaluate model
                    mae = mean_absolute_error(y_test, test_predictions)
                    rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
                    r2 = r2_score(y_test, test_predictions)
                    
                    # Display metrics
                    st.subheader("Model Performance")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Mean Absolute Error", f"{mae:.2f}")
                    col2.metric("Root Mean Squared Error", f"{rmse:.2f}")
                    col3.metric("R² Score", f"{r2:.2f}")
                    
                    # Make future predictions
                    st.subheader("Sales Forecast")
                    
                    # Select number of days to forecast
                    forecast_days = st.slider("Number of days to forecast", min_value=1, max_value=14, value=7)
                    
                    future_dates = pd.date_range(start=daily_sales["Purchase Date"].max(), periods=forecast_days+1, freq='D')[1:]
                    future_predictions = []
                    
                    with st.spinner("Generating forecasts..."):
                        for product_type in daily_sales["Product Type Encoded"].unique():
                            latest_data = daily_sales[daily_sales["Product Type Encoded"] == product_type].iloc[-1:].copy()
                            
                            forecast_features = latest_data[features].copy()
                            
                            for date in future_dates:
                                prediction = model.predict(forecast_features)[0]
                                
                                future_predictions.append({
                                    "Purchase Date": date,
                                    "Product Type": le.inverse_transform([product_type])[0],
                                    "Predicted Quantity": prediction
                                })
                                
                                new_features = forecast_features.copy()
                                for lag in range(7, 1, -1):
                                    new_features[f'lag_{lag}'] = new_features[f'lag_{lag-1}']
                                new_features['lag_1'] = prediction
                                new_features['moving_avg_7'] = new_features[[f'lag_{i}' for i in range(1, 8)]].mean().values[0]
                                
                                forecast_features = new_features
                    
                    forecast_df = pd.DataFrame(future_predictions)
                    
                    st.write("Forecast for the next few days:")
                    st.dataframe(forecast_df)
                    
                    # Create results DataFrame for comparing actual vs predicted values
                    results = pd.DataFrame({
                        'Date': daily_sales.iloc[-len(y_test):]['Purchase Date'],
                        'Product Type': le.inverse_transform(daily_sales.iloc[-len(y_test):]['Product Type Encoded']),
                        'Actual': y_test.values,
                        'Predicted': test_predictions
                    })
                    
                    # Visualize actual vs predicted values
                    st.subheader("Actual vs Predicted Sales")
                    
                    # Overall time-based visualization (total across all products)
                    st.write("### Overall Sales Performance")
                    
                    # Group by date for time-based analysis (sum all products)
                    time_results = results.groupby('Date').agg({
                        'Actual': 'sum',
                        'Predicted': 'sum'
                    }).reset_index()
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Plot actual and predicted values
                    ax.plot(time_results['Date'], time_results['Actual'], label='Actual', marker='o', color='blue')
                    ax.plot(time_results['Date'], time_results['Predicted'], label='Predicted', linestyle='--', marker='x', color='red')
                    
                    # Add text labels for predicted values
                    for i, (date, actual, pred) in enumerate(zip(time_results['Date'], time_results['Actual'], time_results['Predicted'])):
                        ax.annotate(f"{pred:.1f}", 
                                   (date, pred),
                                   textcoords="offset points", 
                                   xytext=(0,10), 
                                   ha='center')
                    
                    # Calculate time-based MAE
                    time_mae = mean_absolute_error(time_results['Actual'], time_results['Predicted'])
                    
                    ax.set_title(f'Total Sales - Actual vs Predicted (MAE: {time_mae:.1f})')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Quantity Sold')
                    ax.legend()
                    ax.grid(True)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Individual product visualizations
                    st.write("### Sales Performance by Product Type")
                    
                    # Get unique product types
                    product_types = results['Product Type'].unique()
                    
                    # Create a multi-select for products or use expanders for all products
                    selected_products = st.multiselect(
                        "Select products to view",
                        options=product_types,
                        default=product_types
                    )
                    
                    if selected_products:
                        # Create columns to display charts side by side (2 columns)
                        cols = st.columns(2)
                        col_idx = 0
                        
                        for product in selected_products:
                            product_data = results[results['Product Type'] == product]
                            
                            # Skip if insufficient data
                            if len(product_data) < 2:
                                continue
                                
                            with cols[col_idx % 2]:
                                fig_p, ax_p = plt.subplots(figsize=(10, 5))
                                
                                # Plot actual and predicted for this product
                                ax_p.plot(product_data['Date'], product_data['Actual'], label='Actual', marker='o', color='blue')
                                ax_p.plot(product_data['Date'], product_data['Predicted'], label='Predicted', linestyle='--', marker='x', color='red')
                                
                                # Add predicted value annotations
                                for i, (date, actual, pred) in enumerate(zip(product_data['Date'], product_data['Actual'], product_data['Predicted'])):
                                    ax_p.annotate(f"{pred:.1f}", 
                                               (date, pred),
                                               textcoords="offset points", 
                                               xytext=(0,10), 
                                               ha='center',
                                               fontsize=8)
                                
                                # Calculate product-specific MAE
                                product_mae = mean_absolute_error(product_data['Actual'], product_data['Predicted'])
                                
                                ax_p.set_title(f'{product} - Actual vs Predicted (MAE: {product_mae:.1f})')
                                ax_p.set_xlabel('Date')
                                ax_p.set_ylabel('Quantity Sold')
                                ax_p.legend()
                                ax_p.grid(True)
                                plt.xticks(rotation=45, fontsize=8)
                                plt.tight_layout()
                                st.pyplot(fig_p)
                            
                            # Move to next column
                            col_idx += 1
                    
                    # Visualize future predictions
                    st.subheader("Sales Forecast")
                    
                    # Group forecast by date (sum across products)
                    time_forecast = forecast_df.groupby('Purchase Date').agg({
                        'Predicted Quantity': 'sum'
                    }).reset_index()
                    
                    fig2, ax2 = plt.subplots(figsize=(12, 6))
                    
                    # Plot the forecast
                    ax2.plot(time_forecast['Purchase Date'], time_forecast['Predicted Quantity'], 
                           marker='o', color='green', label='Forecast')
                    
                    # Add text labels for predicted values
                    for i, (date, pred) in enumerate(zip(time_forecast['Purchase Date'], time_forecast['Predicted Quantity'])):
                        ax2.annotate(f"{pred:.1f}", 
                                   (date, pred),
                                   textcoords="offset points", 
                                   xytext=(0,10), 
                                   ha='center')
                    
                    ax2.set_xlabel("Date")
                    ax2.set_ylabel("Predicted Quantity")
                    ax2.set_title("Total Sales Forecast (All Products)")
                    ax2.grid(True)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    
                    # Individual product forecasts
                    st.write("### Product-Specific Forecasts")
                    
                    # Get unique product types from forecast
                    forecast_products = forecast_df['Product Type'].unique()
                    
                    # Create a multi-select for forecast products
                    selected_forecast_products = st.multiselect(
                        "Select products for forecast view",
                        options=forecast_products,
                        default=forecast_products
                    )
                    
                    if selected_forecast_products:
                        # Create columns to display charts side by side (2 columns)
                        cols = st.columns(2)
                        col_idx = 0
                        
                        for product in selected_forecast_products:
                            product_forecast = forecast_df[forecast_df['Product Type'] == product]
                            
                            with cols[col_idx % 2]:
                                fig_f, ax_f = plt.subplots(figsize=(10, 5))
                                
                                # Plot forecast for this product
                                ax_f.plot(product_forecast['Purchase Date'], product_forecast['Predicted Quantity'], 
                                       marker='o', color='green')
                                
                                # Add predicted value annotations
                                for i, (date, pred) in enumerate(zip(product_forecast['Purchase Date'], product_forecast['Predicted Quantity'])):
                                    ax_f.annotate(f"{pred:.1f}", 
                                               (date, pred),
                                               textcoords="offset points", 
                                               xytext=(0,10), 
                                               ha='center',
                                               fontsize=8)
                                
                                ax_f.set_title(f'{product} - Sales Forecast')
                                ax_f.set_xlabel('Date')
                                ax_f.set_ylabel('Predicted Quantity')
                                ax_f.grid(True)
                                plt.xticks(rotation=45, fontsize=8)
                                plt.tight_layout()
                                st.pyplot(fig_f)
                            
                            # Move to next column
                            col_idx += 1
                    
                    # Download forecast data
                    # Offer both detailed and time-based summary downloads
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv = forecast_df.to_csv(index=False)
                        st.download_button(
                            label="Download detailed forecast data",
                            data=csv,
                            file_name="sales_forecast_detailed.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        time_csv = time_forecast.to_csv(index=False)
                        st.download_button(
                            label="Download time-based forecast summary",
                            data=time_csv,
                            file_name="sales_forecast_summary.csv",
                            mime="text/csv"
                        )
    
    except Exception as e:
        st.error(f"Error processing the file: {str(e)}")
        st.error("Make sure your CSV file has the expected columns: Purchase Date, Product Type, Quantity, Total Price")

else:
    st.info("Please upload a CSV file to begin analysis. The file should contain electronic sales data with columns such as Purchase Date, Product Type, Quantity, and Total Price.")
    
    # Sample data option
    st.write("Need sample data? You can use these columns as a reference:")
    sample_df = pd.DataFrame({
        'Purchase Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'Product Type': ['Smartphone', 'Laptop', 'Headphones'],
        'Quantity': [2, 1, 3],
        'Total Price': [1200.00, 1500.00, 150.00]
    })
    st.dataframe(sample_df)

# Footer
st.markdown("---")
st.markdown("📊 Electronic Sales Analysis & Forecasting App | Created with Streamlit")