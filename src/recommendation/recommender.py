
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
            cols = ["Hospital_Name", "Location_Coordinates", "State", "District", "Specialties", "Hospital_Care_Type"]
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
        """Vectorized Euclidean distance calculation."""
        return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

    def get_recommendations(self, user_lat: float, user_lon: float, disease_category: str = None, top_k: int = 3) -> List[Dict]:
        """
        Find top-K nearest hospitals using vectorized operations.
        """
        if self.df is None or self.df.empty:
            return []

        # 1. Calculate distances for all hospitals at once
        distances = self.calculate_distance(user_lat, user_lon, self.df["lat"], self.df["lon"])
        
        # 2. Add distances to a temporary copy
        temp_df = self.df.copy()
        temp_df["distance"] = distances
        
        # 3. Capability weighting
        temp_df["rank_score"] = temp_df["distance"]
        if disease_category:
            # If specialty or care type matches the category, give a boost
            mask = (temp_df["Specialties"].str.contains(disease_category, case=False, na=False)) | \
                   (temp_df["Hospital_Care_Type"].str.contains(disease_category, case=False, na=False))
            temp_df.loc[mask, "rank_score"] = temp_df["distance"] * 0.5 # 50% better rank
            
        # 4. Sort and return top K
        top_hospitals = temp_df.sort_values("rank_score").head(top_k)
        
        results = []
        for _, row in top_hospitals.iterrows():
            results.append({
                "name": row["Hospital_Name"],
                "city": f"{row['District']}, {row['State']}",
                "specialties": [str(row["Specialties"])] if pd.notna(row["Specialties"]) else ["General"],
                "distance": round(row["distance"], 4)
            })
            
        return results
