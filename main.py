from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Clean column names function
def clean_columns(df):
    df.columns = df.columns.str.replace('\n', ' ')  # Remove newlines in headers
    df.columns = df.columns.str.strip()  # Remove extra spaces
    return df

# Load and preprocess data
try:
    df = pd.read_csv('labour.csv')
    df = clean_columns(df)
    df['Crop Name'] = df['Crop Name'].str.strip()
    print("CSV loaded successfully with columns:\n", df.columns.tolist())
except Exception as e:
    print(f"Error loading CSV: {str(e)}")
    df = None

@app.route('/api/crops', methods=['GET'])
def get_crops():
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
        
    crops = df[['Crop Type', 'Crop Name']].drop_duplicates()
    return jsonify(crops.to_dict(orient='records'))

@app.route('/api/estimate', methods=['POST'])
def estimate_labour():
    if df is None:
        return jsonify({"error": "Dataset not loaded"}), 500

    data = request.get_json()
    
    try:
        crop = data['crop'].strip()
        area = float(data['area'])
        
        if area <= 0:
            raise ValueError("Area must be positive")
            
        crop_data = df[df['Crop Name'].str.strip() == crop]
        
        if crop_data.empty:
            return jsonify({"error": "Crop not found"}), 404
            
        row = crop_data.iloc[0]
        
        def get_rate(prefix, area_range):
            col_name = f"Estimated Cost ({prefix} Rate) - {area_range} ha (In Rs.)"
            return round(row[col_name] * area, 2)
            
        return jsonify({
            "govt_rates": {
                "1_ha": get_rate("Govt.", "1"),
                "2_3_ha": get_rate("Govt.", "2-3"),
                "4_5_ha": get_rate("Govt.", "4-5"),
                "above_5_ha": get_rate("Govt.", "Above 5")
            },
            "expected_rates": {
                "1_ha": get_rate("Expected", "1"),
                "2_3_ha": get_rate("Expected", "2-3"),
                "4_5_ha": get_rate("Expected", "4-5"),
                "above_5_ha": get_rate("Expected", "Above 5")
            },
            "metadata": {
                "crop_type": row['Crop Type'],
                "labour_days": row['Total Labour (per ha)'] * area,
                "sowing_labour": row['Labour for Sowing (per ha)'] * area,
                "harvesting_labour": row['Labour for Harvesting (per ha)'] * area
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
