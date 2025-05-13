from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import os
import re

app = Flask(__name__)

class LabourEstimationModel:
    """Agricultural labour cost estimation model using machine learning."""

    def __init__(self, data=None, csv_path=None, model_path=None):
        """Initialize the model, load data, generate synthetic data, and train models."""
        self.df = None
        self.scaling_factors = {
            "small": 1.0,
            "medium": 2.5,
            "large": 4.5,
            "xlarge": 6.0
        }
        self.efficiency_factors = {
            "small": 1.0,
            "medium": 0.95,
            "large": 0.9,
            "xlarge": 0.85
        }

        if csv_path and os.path.exists(csv_path):
            self.load_from_csv(csv_path)
        elif data:
            self.initialize_data(data)
        else:
            self.initialize_default_data()

        self.generate_synthetic_data()
        self.preprocess_data()
        self.train_models()

    def initialize_default_data(self):
        """Initialize with default crop data."""
        data = {
            "Crop Type": ["Vegetable", "Vegetable", "Vegetable"],
            "Crop Name": ["Tomato", "Potato", "Onion"],
            "Govt. Fixed Wage Rate (rs/day)": [350, 350, 350],
            "Expected Wage Rate (rs/day)": [450, 450, 450],
            "Labour for Sowing (per ha)": [12, 10, 15],
            "Labour for Harvesting (per ha)": [20, 18, 25],
            "Total Labour (per ha)": [32, 28, 40]
        }
        self.df = pd.DataFrame(data)

    def initialize_data(self, data):
        """Initialize with provided data."""
        self.df = pd.DataFrame(data)
        if "Total Labour (per ha)" not in self.df.columns:
            self.df["Total Labour (per ha)"] = self.df["Labour for Sowing (per ha)"] + self.df["Labour for Harvesting (per ha)"]

    def load_from_csv(self, filepath):
        """Load crop data from a CSV file."""
        try:
            self.df = pd.read_csv(filepath, sep=',', header=0, encoding='latin-1')
            self.df.columns = [re.sub(r'\s+', ' ', col.replace('\n', ' ').replace('"', '')).strip() for col in self.df.columns]
            required_columns = [
                "Crop Type", "Crop Name", "Govt. Fixed Wage Rate (rs/day)",
                "Expected Wage Rate (rs/day)", "Labour for Sowing (per ha)",
                "Labour for Harvesting (per ha)", "Total Labour (per ha)"
            ]
            missing = [col for col in required_columns if col not in self.df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            if "Total Labour (per ha)" not in self.df.columns:
                self.df["Total Labour (per ha)"] = self.df["Labour for Sowing (per ha)"] + self.df["Labour for Harvesting (per ha)"]
        except Exception as e:
            raise RuntimeError(f"Error loading CSV: {e}")

    def get_farm_size_category(self, area):
        """Determine farm size based on area."""
        if area <= 1:
            return "small"
        elif area <= 3:
            return "medium"
        elif area <= 5:
            return "large"
        else:
            return "xlarge"

    def generate_synthetic_data(self):
        """Generate synthetic training data with seasonality and varying areas."""
        labor_demand_samples = []
        labor_cost_samples = []
        seasons = {'Spring': 1.0, 'Summer': 1.1, 'Fall': 0.95, 'Winter': 0.9}

        for _, row in self.df.iterrows():
            crop_type = row['Crop Type']
            govt_wage = row['Govt. Fixed Wage Rate (rs/day)']
            expected_wage = row['Expected Wage Rate (rs/day)']
            labor_per_ha = row['Total Labour (per ha)']

            for area in np.arange(0.5, 10.5, 0.5):
                farm_size = self.get_farm_size_category(area)
                efficiency = self.efficiency_factors[farm_size]

                for season, season_factor in seasons.items():
                    adj_labor_per_ha = labor_per_ha * efficiency * season_factor
                    labor_demand = adj_labor_per_ha * area
                    labor_demand_samples.append({
                        'Crop Type': crop_type,
                        'Area': area,
                        'Season': season,
                        'Labor Demand': labor_demand
                    })

                    for wage_type in ['Govt', 'Expected']:
                        wage = govt_wage if wage_type == 'Govt' else expected_wage
                        cost_per_ha = adj_labor_per_ha * wage
                        labor_cost_samples.append({
                            'Crop Type': crop_type,
                            'Area': area,
                            'Season': season,
                            'Labor Required per ha': adj_labor_per_ha,
                            'Govt Wage Rate': govt_wage,
                            'Expected Wage Rate': expected_wage,
                            'Wage Type': wage_type,
                            'Cost per ha': cost_per_ha
                        })

        self.labor_demand_data = pd.DataFrame(labor_demand_samples)
        self.labor_cost_data = pd.DataFrame(labor_cost_samples)

    def preprocess_data(self):
        """Preprocess data for model training."""
        self.encoder_demand = OneHotEncoder(handle_unknown='ignore')
        demand_cat = self.labor_demand_data[['Crop Type', 'Season']]
        self.encoder_demand.fit(demand_cat)
        encoded_demand = self.encoder_demand.transform(demand_cat).toarray()
        self.demand_X = np.hstack([encoded_demand, self.labor_demand_data[['Area']].values])
        self.demand_y = self.labor_demand_data['Labor Demand'].values

        self.encoder_cost = OneHotEncoder(handle_unknown='ignore')
        cost_cat = self.labor_cost_data[['Crop Type', 'Season', 'Wage Type']]
        self.encoder_cost.fit(cost_cat)
        encoded_cost = self.encoder_cost.transform(cost_cat).toarray()
        self.cost_X = np.hstack([
            encoded_cost,
            self.labor_cost_data[['Area', 'Labor Required per ha', 'Govt Wage Rate', 'Expected Wage Rate']].values
        ])
        self.cost_y = self.labor_cost_data['Cost per ha'].values

    def train_models(self):
        """Train the machine learning models."""
        self.demand_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.demand_model.fit(self.demand_X, self.demand_y)

        self.cost_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.cost_model.fit(self.cost_X, self.cost_y)

    def estimate_labor_demand(self, crop_type, area, season):
        """Predict the number of laborers required."""
        input_features = pd.DataFrame([[crop_type, season]], columns=['Crop Type', 'Season'])
        encoded = self.encoder_demand.transform(input_features).toarray()
        features = np.hstack([encoded, [[area]]])
        return self.demand_model.predict(features)[0]

    def estimate_cost(self, crop_name, area, wage_type='Govt', season='Spring'):
        """Predict the total labor cost and cost per hectare."""
        crop_row = self.df[self.df['Crop Name'] == crop_name]
        if crop_row.empty:
            return {"error": f"Crop '{crop_name}' not found."}

        crop_type = crop_row['Crop Type'].values[0]
        govt_wage = crop_row['Govt. Fixed Wage Rate (rs/day)'].values[0]
        expected_wage = crop_row['Expected Wage Rate (rs/day)'].values[0]

        labor_demand = self.estimate_labor_demand(crop_type, area, season)
        labor_per_ha = labor_demand / area

        input_cost = pd.DataFrame([[
            crop_type, season, wage_type, area, labor_per_ha, govt_wage, expected_wage
        ]], columns=[
            'Crop Type', 'Season', 'Wage Type', 'Area', 'Labor Required per ha',
            'Govt Wage Rate', 'Expected Wage Rate'
        ])

        encoded_cost = self.encoder_cost.transform(input_cost[['Crop Type', 'Season', 'Wage Type']]).toarray()
        cost_features = np.hstack([
            encoded_cost,
            input_cost[['Area', 'Labor Required per ha', 'Govt Wage Rate', 'Expected Wage Rate']].values
        ])

        cost_per_ha = self.cost_model.predict(cost_features)[0]
        total_cost = cost_per_ha * area

        return {
            "crop": crop_name,
            "area": area,
            "season": season,
            "wage_type": wage_type,
            "total_cost": round(total_cost, 2),
            "cost_per_ha": round(cost_per_ha, 2),
            "labor_demand": round(labor_demand)
        }

# Initialize model with default data
model = LabourEstimationModel()

@app.route('/', methods=['GET', 'POST'])
def index():
    crops = model.df['Crop Name'].unique().tolist()
    seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    wage_types = ['Govt', 'Expected']
    areas = [round(0.5 * i, 1) for i in range(1, 21)]  # 0.5 to 10.0 in 0.5 increments
    
    return render_template('index.html',
                           crops=crops,
                           seasons=seasons,
                           wage_types=wage_types,
                           areas=areas)

@app.route('/estimate', methods=['POST'])
def estimate():
    crop_name = request.form['crop_name']
    area = float(request.form['area'])
    season = request.form['season']
    wage_type = request.form['wage_type']
    
    result = model.estimate_cost(crop_name, area, wage_type, season)
    
    crops = model.df['Crop Name'].unique().tolist()
    seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    wage_types = ['Govt', 'Expected']
    areas = [round(0.5 * i, 1) for i in range(1, 21)]
    
    return render_template('index.html',
                           crops=crops,
                           seasons=seasons,
                           wage_types=wage_types,
                           areas=areas,
                           result=result,
                           selected_crop=crop_name,
                           selected_area=area,
                           selected_season=season,
                           selected_wage=wage_type)

if __name__ == '__main__':
    app.run(debug=True)