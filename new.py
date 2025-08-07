from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pandas as pd
import os

app = Flask(__name__)

# Enable CORS globally
CORS(app)

def load_data():
    csv_path = os.environ.get('LABOUR_CSV_PATH', 'labour.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at path: {csv_path}")

    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [
        c.strip().replace('\n', '').replace(' ', '_').replace('.', '').replace('(', '').replace(')', '')
        for c in df.columns
    ]
    return df

try:
    df = load_data()
except Exception as e:
    print(f"Error loading CSV data: {e}")
    df = None  # Optionally exit app here if critical

def get_cost_column(area, wage_type):
    area = float(area)
    if area <= 1:
        return f"Estimated_Cost_{wage_type}_Rate_-_1_haIn_Rs"
    elif 2 <= area <= 3:
        return f"Estimated_Cost_{wage_type}_Rate_-_2-3_haIn_Rs"
    elif 4 <= area <= 5:
        return f"Estimated_Cost_{wage_type}_Rate_-_4-5_haIn_Rs"
    else:
        return f"Estimated_Cost_{wage_type}_Rate_-_Above_5_haIn_Rs"


@app.route('/api/labour-estimate', methods=['POST', 'OPTIONS'])
@cross_origin()
def api_labour_estimate():
    if df is None:
        return jsonify({"error": "Server data not loaded."}), 500

    if request.method == 'OPTIONS':
        # Explicit handling of OPTIONS preflight request
        response = app.make_response('')
        response.status_code = 204
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided."}), 400

    try:
        crop_name = data['crop_name']
        area = float(data['area'])
        wage_type = data['wage_type']
        season = data.get('season', 'N/A')
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400

    wage_type_col = "Govt" if wage_type.lower().startswith("govt") else "Expected"
    row = df[df['Crop_Name'].str.lower() == crop_name.lower()]
    if row.empty:
        return jsonify({"error": "Crop not found."}), 404

    total_labour_per_ha = float(row.iloc[0]['Total_Labour_per_ha'])
    cost_col = get_cost_column(area, wage_type_col)
    if cost_col not in row.columns:
        return jsonify({"error": "Cost column not found for this area/wage type."}), 400

    cost_per_hectare = float(row.iloc[0][cost_col])

    if area in [1, 2, 3, 4, 5]:
        total_cost = cost_per_hectare
    else:
        per_ha_col = f"Estimated_Cost_{wage_type_col}_Rate_-_1_haIn_Rs"
        per_ha_cost = float(row.iloc[0][per_ha_col])
        total_cost = per_ha_cost * area

    return jsonify({
        "crop": crop_name,
        "area": area,
        "season": season,
        "wage_type": wage_type,
        "total_labour_per_ha": total_labour_per_ha,
        "cost_per_hectare": cost_per_hectare,
        "total_cost": round(total_cost, 2)
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
