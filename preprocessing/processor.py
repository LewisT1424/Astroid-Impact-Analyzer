import polars as pl
import numpy as np
from sklearn.preprocessing import RobustScaler
from utils import logger

class Preprocessor:
    def __init__(self):
        self.data = pl.read_parquet('data/cleaned/data_2025-07-08.parquet')

    def format_string_cols(self):
        '''
        This function deals with string columns
        '''
        time_period_order = {
            'night': 0,
            'morning': 1,
            'afternoon': 2,
            'evening': 3
        }
        df = self.data.with_columns([
            # Date columns
            pl.col('date').str.to_date().alias('date_parsed'),
            pl.col('close_approach_date').str.to_datetime("%Y-%b-%d %H:%M").alias('approach_date_parsed')
        ]).with_columns([
            # Create time-based features
                (pl.col('approach_date_parsed').dt.hour() * 60 + pl.col('approach_date_parsed').dt.minute()).abs().alias('minutes_since_midnight'),

                pl.when(pl.col('approach_date_parsed').dt.hour().is_between(6, 11))
                .then(pl.lit('morning'))
                .when(pl.col('approach_date_parsed').dt.hour().is_between(12, 17))
                .then(pl.lit('afternoon'))
                .when(pl.col('approach_date_parsed').dt.hour().is_between(18, 21))
                .then(pl.lit('evening'))
                .otherwise(pl.lit('night')).alias('time_period')
        ]).with_columns(
            pl.col('time_period').replace(time_period_order).cast(int).alias('time_period_encoded')
        ).drop('time_period')



        # Patterns from identifiers
        df = df.with_columns([
            pl.col('name').str.len_chars().alias('name_length'),
            pl.col('name').str.extract(r"\((\d{4})").cast(pl.Int32, strict=False).alias('discovery_year'),
            pl.col('id').cast(pl.Int64, strict=False).alias('id')
        ])

        # Handle orbiting_body 
        if len(df['oribiting_body'].unique()) == 1:
            df = df.drop('oribiting_body')
        else:
            df=  df.to_dummies(columns=['oribiting_body'], separator='_')

        # Oridinal encoding for categories
        size_order = {'tiny': 0, 'small': 1, 'medium': 2, 'large': 3}
        velocity_order = {"very_fast": 3, "slow": 0, "medium": 1, "fast": 2}
        proximity_order = {"moderate": 1,	"close": 2, "very_close": 3 ,	"far": 0}
        brightness_order = {'very_dim': 0, 'dim': 1, 'bright': 2, 'very_bright': 3}

        # Map df cols to encoding above
        df = df.with_columns([
            pl.col('size_category').replace(size_order).cast(pl.Float32).alias('size_encoded'),
            pl.col('velocity_category').replace(velocity_order).cast(pl.Float32).alias('velocity_encoded'),
            pl.col('proximity_level').replace(proximity_order).cast(pl.Float32).alias('proximity_encoded'),
            pl.col('brightness_category').replace(brightness_order).cast(pl.Float32).alias('brightness_encoded'),
            pl.col('is_sentry_object').replace({True: 1, False: 0}).cast(int).alias('is_sentry_object'),
            pl.col('is_potentially_hazardous').replace({True: 1, False: 0}).cast(int).alias('is_potentially_hazardous')        
        ])

        # Remove cols from df 
        strings_cols_to_drop = ['date', 'close_approach_date', 'date_parsed', 'approach_date_parsed', 'id', 'name', 'size_category', 'velocity_category', 'proximity_level', 'brightness_category']
        exisiting_drops = [col for col in strings_cols_to_drop if col in df.columns]
        df = df.drop(exisiting_drops).rename({'relative_velocity_km/sec': 'relative_velocity_km_sec',	'relative_velocity_km/hr':'relative_velocity_km_hr'})

        return df
    
    def final_data_transformation(self):
        df = self.data.to_pandas().copy()
        # Zero-heavy features (apply log1p)
        zero_heavy_features = [
            'momentum', 'size_squared_velocity', 'velocity_distance_ratio',
            'velocity_per_au', 'size_velocity_product', 'apparent_density_inverse',
            'impact_potential'
        ]
        
        # Skewed features
        skewed_features = [
            'threat_score', 'name_length', 'obj_that_day', 'brightness_size_ratio',
            'close_approach_score', 'hazard_index'
        ]
        
        # Diameter features (all highly skewed)
        diameter_features = [
            'estimated_diameter_min_km', 'estimated_diameter_max_km',
            'estimated_diameter_min_m', 'estimated_diameter_max_m',
            'estimated_diameter_min_miles', 'estimated_diameter_max_miles',
            'estimated_diameter_min_feet', 'estimated_diameter_max_feet',
            'avg_diameter_km', 'diameter_uncertainty_km'
        ]
        
        # Scientific features
        scientific_features = [
            'estimated_volume', 'cross_section_area_km2', 'kenetic_energy', 
            'destruction_potential'
        ]
        
        # Apply log1p transformation
        all_log_features = zero_heavy_features + skewed_features + diameter_features + scientific_features
        
        log_count = 0
        for feature in all_log_features:
            if feature in df.columns:
                # Check if already log-transformed
                if f'log_{feature}' not in df.columns:
                    df[f'log_{feature}'] = np.log1p(df[feature].clip(lower=0))
                    log_count += 1
        
        logger.info(f"Applied log transformation to {log_count} features")


        cyclical_features = {
            'approach_month': 12,
            'day_of_week': 7,
            'day_of_year': 365,
            'approach_hour': 24
        }
        
        cyclical_count = 0
        for feature, max_val in cyclical_features.items():
            if feature in df.columns:
                # Only add if not already present
                if f'{feature}_sin' not in df.columns:
                    df[f'{feature}_sin'] = np.sin(2 * np.pi * df[feature] / max_val)
                    df[f'{feature}_cos'] = np.cos(2 * np.pi * df[feature] / max_val)
                    cyclical_count += 1
        
        logger.info(f"   Added cyclical encoding for {cyclical_count} features")

        features_to_drop = [
            # Redundant diameter features (keep only log_estimated_diameter_min_km)
            'estimated_diameter_min_km', 'estimated_diameter_max_km',
            'estimated_diameter_min_m', 'estimated_diameter_max_m',
            'estimated_diameter_min_miles', 'estimated_diameter_max_miles',
            'estimated_diameter_min_feet', 'estimated_diameter_max_feet',
            'avg_diameter_km', 'diameter_uncertainty_km',
            
            # Keep only one log diameter feature
            'log_estimated_diameter_max_km', 'log_estimated_diameter_min_m',
            'log_estimated_diameter_max_m', 'log_estimated_diameter_min_miles',
            'log_estimated_diameter_max_miles', 'log_estimated_diameter_min_feet',
            'log_estimated_diameter_max_feet', 'log_avg_diameter_km',
            'log_diameter_uncertainty_km',
            
            # Redundant velocity features (keep relative_velocity_km_sec)
            'relative_velocity_km_hr', 'relative_velocity_mph',
            
            # Redundant distance features (keep miss_distance_astronomical)
            'miss_distance_lunar', 'miss_distance_kilometers', 'miss_distance_miles',
            
            # Original features that were log-transformed
            'momentum', 'size_squared_velocity', 'velocity_distance_ratio',
            'velocity_per_au', 'size_velocity_product', 'apparent_density_inverse',
            'impact_potential', 'threat_score', 'brightness_size_ratio',
            'close_approach_score', 'hazard_index', 'estimated_volume',
            'cross_section_area_km2', 'kenetic_energy', 'destruction_potential',
            
            # Low-quality features
            'diameter_uncertainty_ratio', 'distance_zscore', 'discovery_year',
            
            # Temporal features we don't need
            'epoch_date_close_approach', 'approach_datetime',
            
            # Keep only one of each cyclical encoding pair if duplicated
            'approach_month_cos',  # Keep sin version
            
            # String/ID columns
            'id', 'name', 'date', 'close_approach_date', 'oribiting_body',
            'size_category', 'velocity_category', 'proximity_level', 'brightness_category'
        ]
        
        # Remove features that exist in the dataframe
        features_dropped = []
        for feature in features_to_drop:
            if feature in df.columns:
                df = df.drop(columns=[feature])
                features_dropped.append(feature)
        
        logger.info(f"   Dropped {len(features_dropped)} redundant features")

        if 'is_potentially_hazardous' in df.columns:
            y = df['is_potentially_hazardous'].copy()
            df = df.drop(columns=['is_potentially_hazardous'])
            logger.info(f"   Target variable extracted. Class distribution:")
            logger.info(f"   Non-hazardous: {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
            logger.info(f"   Hazardous: {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")
        else:
            logger.info("   ⚠️ Warning: 'is_potentially_hazardous' not found")
            y = None

        # Keep only numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        df = df[numeric_columns]
        
        # Remove any remaining highly correlated features
        corr_matrix = df.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation > 0.95
        high_corr_features = [column for column in upper_triangle.columns 
                            if any(upper_triangle[column] > 0.95)]
        
        if high_corr_features:
            df = df.drop(columns=high_corr_features[:len(high_corr_features)//2])  # Drop half
            logger.info(f"   Removed {len(high_corr_features)//2} highly correlated features")

        no_scale_features = [
            col for col in df.columns if any(keyword in col.lower() for keyword in 
            ['sin', 'cos', 'encoded', 'percentile', 'zscore', 'is_'])
        ]
        
        # Features to scale
        features_to_scale = [col for col in df.columns if col not in no_scale_features]
        
        if features_to_scale:
            # Use RobustScaler (better for outliers than StandardScaler)
            scaler = RobustScaler()
            df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
            logger.info(f"   Scaled {len(features_to_scale)} features using RobustScaler")
        else:
            scaler = None
            logger.info("   No features needed scaling")

        # Remove any columns with all zeros or constant values
        constant_columns = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_columns:
            df = df.drop(columns=constant_columns)
            logger.info(f"   Removed {len(constant_columns)} constant columns")
        
        # Ensure no infinite or extremely large values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # Clip extreme outliers (beyond 5 standard deviations)
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].std() > 0:  # Only for non-constant columns
                mean_val = df[col].mean()
                std_val = df[col].std()
                df[col] = df[col].clip(mean_val - 5*std_val, mean_val + 5*std_val)

        logger.info("\n✅ Transformation Complete!")
        logger.info(f"📊 Final shape: {df.shape}")
        logger.info(f"🎯 Features: {df.shape[1]}")
        logger.info(f"📝 Samples: {df.shape[0]}")
        
        if y is not None:
            logger.info(f"⚖️ Class ratio: {(y==0).sum()/(y==1).sum():.1f}:1")
        
        # Show final feature list
        feature_names = df.columns.tolist()
        logger.info(f"\n📋 Final features ({len(feature_names)}):")
        for i, feature in enumerate(feature_names, 1):
            logger.info(f"   {i:2d}. {feature}")
        
        # Show sample statistics
        logger.info(f"\n📈 Sample feature statistics:")
        logger.info(df.describe().iloc[:, :5])  # Show first 5 columns
        
        return df, y, feature_names, scaler


    def run_pipeline(self):
        string_transformed_data = self.format_string_cols(self.data)
        data, target, feature_names, scaler = self.final_data_transformation(string_transformed_data)

        return data, target, feature_names, scaler
    
def main():
    preprocessor = Preprocessor()

    return preprocessor.run_pipeline()

if __name__ == '__main__':
    main()