
import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class HospitalRecommender:
    """Proximity and capability based recommendation engine using the National Hospital Directory."""
    
    def __init__(self, csv_path: str = None):
        if csv_path is None:
            csv_path = str(Path(__file__).parents[2] / "data" / "raw" / "hospitals" / "hospital_directory.csv")
        
        self.csv_path = csv_path
        self.df = None
        self.load_data()

    def load_data(self):
        """Load and preprocess the hospital directory CSV."""
        try:
            # Loading with specific columns to save memory
            cols = ["Hospital_Name", "Location_Coordinates", "State", "District", "Specialties", "Hospital_Care_Type", "Emergency_24x7"]
            self.df = pd.read_csv(self.csv_path, usecols=cols)
            
            # Clean coordinates
            self.df = self.df.dropna(subset=["Location_Coordinates"])
            
            # Parse Lat/Lon
            coords = self.df["Location_Coordinates"].str.split(",", expand=True)
            self.df["lat"] = pd.to_numeric(coords[0], errors='coerce')
            self.df["lon"] = pd.to_numeric(coords[1], errors='coerce')
            
            # Drop invalid coordinates
            self.df = self.df.dropna(subset=["lat", "lon"])
            
            logger.info(f"Loaded {len(self.df)} hospitals from directory.")
        except Exception as e:
            logger.error(f"Error loading hospital directory: {e}")
            self.df = pd.DataFrame()

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate Haversine distance in kilometers."""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371.0 # Earth mean radius in km
        return c * r

    def get_recommendations(self, user_lat: float, user_lon: float, disease_category: str = None, top_k: int = 3) -> List[Dict]:
        """
        Find top-K nearest hospitals using vectorized operations.
        Validates user coordinates and handles missing/out-of-bound inputs.
        """
        if self.df is None or self.df.empty:
            return []

        # Validate coordinates; fallback to Bangalore city center if invalid
        if user_lat is None or user_lon is None or not (-90.0 <= user_lat <= 90.0 and -180.0 <= user_lon <= 180.0):
            logger.warning(f"Invalid user coordinates ({user_lat}, {user_lon}); defaulting to Bangalore Center (12.9716, 77.5946)")
            user_lat, user_lon = 12.9716, 77.5946

        # 1. Calculate distances for all hospitals at once
        distances = self.calculate_distance(user_lat, user_lon, self.df["lat"], self.df["lon"])
        
        # 2. Add distances to a temporary copy
        temp_df = self.df.copy()
        temp_df["distance"] = distances
        
        # 3. Capability weighting
        temp_df["rank_score"] = temp_df["distance"]
        if disease_category:
            # Clean category string and use literal substring matching (regex=False)
            clean_category = str(disease_category).strip()
            mask = (temp_df["Specialties"].str.contains(clean_category, case=False, na=False, regex=False)) | \
                   (temp_df["Hospital_Care_Type"].str.contains(clean_category, case=False, na=False, regex=False))
            temp_df.loc[mask, "rank_score"] = temp_df["distance"] * 0.5 # 50% better rank
            
        # 4. Sort and return top K
        top_hospitals = temp_df.sort_values("rank_score").head(top_k)
        
        results = []
        for _, row in top_hospitals.iterrows():
            results.append({
                "name": row["Hospital_Name"],
                "city": f"{row['District']}, {row['State']}",
                "specialties": [str(row["Specialties"])] if pd.notna(row["Specialties"]) else ["General"],
                "emergency_24x7": str(row.get("Emergency_24x7", "Yes")),
                "distance": round(row["distance"], 2)
            })
            
        return results
