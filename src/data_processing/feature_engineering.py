import polars as pl
import json
import numpy as np
import os
from src.utils.utils import logger, save_df_to_pqt
from typing import List

class Preprocessor:
    def __init__(self):
        self.files = os.listdir('data/raw')

    def collect_data(self) -> List:  
        results = []
        for idx, file_name in enumerate(self.files):
            try:
                with open(f"data/raw/{file_name}", 'r') as f:
                    data = json.load(f)
                    results.append(data)
                    logger.info(f"Data successfully loaded: {idx+1}/{len(self.files)}")

            except Exception as e:
                logger.error(f"Error occured reading data from json file:{file_name}: {e}")
        return results
    
    def format_data(self, data) -> pl.DataFrame:
        data_list = []
        try:
            for date in data['near_earth_objects']:
                result = {'date': date}
                for idx, obj in enumerate(data['near_earth_objects'][date]):
                    new_result = result | {
                        # General stuff about astroid
                        'obj_that_day': idx,
                        'id': str(obj['id']),
                        'name': obj['name'],
                        'absolute_magniutude_h': obj['absolute_magnitude_h'],
                        # Estimated Diameter
                        'estimated_diameter_min_km': obj['estimated_diameter']['kilometers']['estimated_diameter_min'],
                        'estimated_diameter_max_km': obj['estimated_diameter']['kilometers']['estimated_diameter_max'],
                        'estimated_diameter_min_m': obj['estimated_diameter']['meters']['estimated_diameter_min'],
                        'estimated_diameter_max_m': obj['estimated_diameter']['meters']['estimated_diameter_max'],
                        'estimated_diameter_min_miles': obj['estimated_diameter']['miles']['estimated_diameter_min'],
                        'estimated_diameter_max_miles': obj['estimated_diameter']['miles']['estimated_diameter_max'],
                        'estimated_diameter_min_feet': obj['estimated_diameter']['feet']['estimated_diameter_min'],
                        'estimated_diameter_max_feet': obj['estimated_diameter']['feet']['estimated_diameter_max'],
                        # Potentially hazardous, Sentry object refers to if the astroid is tracked by nasa's sentry system 
                        'is_potentially_hazardous': obj['is_potentially_hazardous_asteroid'],
                        'is_sentry_object': obj['is_sentry_object']
                    }
                    for val in obj['close_approach_data']:
                        final_result = new_result | {
                            # Close approaching dates
                            'close_approach_date': val['close_approach_date_full'],
                            'epoch_date_close_approach': val['epoch_date_close_approach'],
                            # Velocity values
                            'relative_velocity_km/sec': float(val['relative_velocity']['kilometers_per_second']),
                            'relative_velocity_km/hr': float(val['relative_velocity']['kilometers_per_hour']),
                            'relative_velocity_mph': float(val['relative_velocity']['miles_per_hour']),
                            # Miss distance
                            'miss_distance_astronomical': float(val['miss_distance']['astronomical']),
                            'miss_distance_lunar': float(val['miss_distance']['lunar']),
                            'miss_distance_kilometers': float(val['miss_distance']['kilometers']),
                            'miss_distance_miles': float(val['miss_distance']['miles']),
                            # Orbiting body 
                            'oribiting_body': val['orbiting_body']
                        }

                        data_list.append(final_result)
                        logger.info(f"Data added to result")

            return pl.DataFrame(data_list)
        except Exception as e:
            logger.error(f"Error with the amount of near_earth_objects: {e}")

    def feature_engineering(self, data: pl.DataFrame) -> pl.DataFrame:
        df = data.with_columns([
            # Size based features
            ((pl.col('estimated_diameter_min_km') + pl.col('estimated_diameter_max_km')) / 2).alias('avg_diameter_km'),
            (pl.col('estimated_diameter_max_km') - pl.col('estimated_diameter_min_km')).alias('diameter_uncertainty_km'),
            
            # Volume of sphere (4/3 x pi x r^3)
            ((4/3) * np.pi * (((pl.col('estimated_diameter_min_km') + pl.col('estimated_diameter_max_km')) / 4) ** 3)).alias('estimated_volume'),
            # Cross section area from earth (pi x r^2)
            (np.pi * (((pl.col('estimated_diameter_min_km') + pl.col('estimated_diameter_max_km')) / 4) ** 2)).alias('cross_section_area_km2')
        
        ]).with_columns([
            # Diameter uncertentity ratio 
            (pl.col('diameter_uncertainty_km') / pl.col('avg_diameter_km')).alias('diameter_uncertainty_ratio'),
            # Site category
            pl.when(pl.col('avg_diameter_km') < 0.05).then(pl.lit('tiny'))
            .when(pl.col('avg_diameter_km') < 0.14).then(pl.lit('small'))
            .when(pl.col('avg_diameter_km') < 0.5).then(pl.lit('medium'))
            .otherwise(pl.lit('large')).alias('size_category'),

            # Velocity & Kenetic energy features
            # Kenetic Energy (KE = 1/2 * m * v^2)
            (pl.col('estimated_volume') * pl.col('relative_velocity_km/sec') ** 2).alias('kenetic_energy'),
            # Momentum (volume x velocity)
            (pl.col('estimated_volume') * pl.col('relative_velocity_km/sec')).alias('momentum'),
            # velocity per atronomic unit
            (pl.col('relative_velocity_km/sec') / pl.col('miss_distance_astronomical')).alias('velocity_per_au'),
            # velocity distance ratio (fast + close = dangerous, slow + far = safe)
            (pl.col('relative_velocity_km/sec') / (pl.col('miss_distance_kilometers')/ 1e6)).alias('velocity_distance_ratio'),
            # Velocity category
            pl.when(pl.col('relative_velocity_km/sec') < 10).then(pl.lit('slow'))
            .when(pl.col('relative_velocity_km/sec') < 20).then(pl.lit('medium'))
            .when(pl.col('relative_velocity_km/sec') < 30).then(pl.lit('fast'))
            .otherwise(pl.lit('very_fast')).alias('velocity_category'),

        ]).with_columns([
            # Distance & Risk factors
            # Lunar distance ratio
            (pl.col('miss_distance_lunar') / 384400).alias("lunar_distance_ratio"),
            # Earth radii distance (earth in radius unites)
            (pl.col('miss_distance_kilometers')/ 6371).alias('earth_radii_distance'),
            # Close approach score (closer = higher score)
            (1 / pl.col('miss_distance_astronomical')).alias('close_approach_score'),
            # Impact potential (combined size, speed and proximity threat)
            ((pl.col('avg_diameter_km') * pl.col('relative_velocity_km/sec')) / pl.col('miss_distance_astronomical')).alias('impact_potential'),
            # Destruction potential (energy density at earths distance)
            (pl.col('kenetic_energy') / pl.col('miss_distance_astronomical')).alias('destruction_potential'),
            # Hazard index (combined size^2, velocity^2 and distance): (diameter^2 x velocity^2) / distance_km
            ((pl.col('avg_diameter_km') ** 2 * pl.col('relative_velocity_km/sec') ** 2) / pl.col('miss_distance_kilometers')).alias('hazard_index'),
            # Proximity level
            pl.when(pl.col('miss_distance_lunar') < 10).then(pl.lit('very_close'))
            .when(pl.col('miss_distance_lunar') < 50).then(pl.lit('close'))
            .when(pl.col('miss_distance_lunar') < 100).then(pl.lit('moderate'))
            .otherwise(pl.lit('far')).alias('proximity_level'),
            
            # Approach datetime - Convert epoch timestamp to readable datetime
            pl.from_epoch('epoch_date_close_approach', time_unit='ms').alias('approach_datetime')

        ]).with_columns([
            # Temporal features
            # Time components
            pl.col('approach_datetime').dt.year().alias('approach_year'),
            pl.col('approach_datetime').dt.month().alias('approach_month'),
            pl.col('approach_datetime').dt.day().alias('approach_day'),
            pl.col('approach_datetime').dt.hour().alias('approach_hour'),

            # Day of week and year
            pl.col('approach_datetime').dt.weekday().alias('day_of_week'),
            pl.col('approach_datetime').dt.ordinal_day().alias('day_of_year')
    
        ]).with_columns([
            # Month_sin/month_cos + hour_sin/hour_cos (cyclical encoding of moinths)
            (2 * np.pi * pl.col('approach_month') / 12).sin().alias('month_sin'),
            (2 * np.pi * pl.col('approach_month') / 12).cos().alias('month_cos'),
            (2 * np.pi * pl.col('approach_hour') / 24).sin().alias('hour_sin'),
            (2 * np.pi * pl.col('approach_hour') / 24).cos().alias('hour_cos'),

            # Brightness & physical features
            # Brightness size ratio
            (pl.col('absolute_magniutude_h') / pl.col('avg_diameter_km')).alias('brightness_size_ratio'),
            # Apparant densitiy inversed (1/volume) (higher = denser)
            # Dense (metal) vs Fluffy (rubble pile)
            (1 / pl.col('estimated_volume')).alias('apparent_density_inverse'),

            # Brightness category
            pl.when(pl.col('absolute_magniutude_h') < 20).then(pl.lit('very_bright'))
            .when(pl.col('absolute_magniutude_h') < 22).then(pl.lit('bright'))
            .when(pl.col('absolute_magniutude_h') < 25).then(pl.lit('dim'))
            .otherwise(pl.lit('very_dim')).alias('brightness_category'),

            # Interaction features

            # Size velocity produt (size and speed)
            (pl.col('avg_diameter_km') * pl.col('relative_velocity_km/sec')).alias("size_velocity_product"),
            # Size squared velocity (emphasises on size) (diameter^2 x velocity)
            (pl.col('avg_diameter_km') ** 2 * pl.col('relative_velocity_km/sec')).alias("size_squared_velocity"),
            # Escape velocity ratio (speed compared to easrth escape velocity)
            (pl.col('relative_velocity_km/sec') / 11.2).alias('escape_velocity_ratio'),
            # Threat score (size x velocity) / (distance + weighed_value_to_avoid_div_0)
            ((pl.col('avg_diameter_km') * pl.col('relative_velocity_km/sec')) / (pl.col('miss_distance_astronomical') + 0.001)).alias('threat_score'),

            # Normalisation & Scaling features
            # Size percentile (0-1)
            (pl.col('avg_diameter_km').rank(method='average') / pl.col('avg_diameter_km').len()).alias('size_percentile'),
            # Velocity percentile
            (pl.col('relative_velocity_km/sec').rank(method='average') / pl.col('relative_velocity_km/sec').len()).alias('velocity_percentile'),
            # Distance percentile
            (pl.col('miss_distance_astronomical').rank(method='average') / pl.col('miss_distance_astronomical').len()).alias('distance_percentile')
        ]).with_columns([
            # Z-scores & Log values
            # Size zscore
            ((pl.col('avg_diameter_km') - pl.col('avg_diameter_km').mean()) / pl.col('avg_diameter_km').std()).alias('size_zscore'),
            # Velocity zscore
            ((pl.col('relative_velocity_km/sec') - pl.col('relative_velocity_km/sec').mean()) / pl.col('relative_velocity_km/sec').std()).alias('velocity_zscore'),
            # distance zscore
            ((pl.col('miss_distance_astronomical') - pl.col('miss_distance_astronomical').mean()) / pl.col('miss_distance_astronomical').std()).alias('distance_zscore'),

            # Log diameter 
            pl.col('avg_diameter_km').log1p().alias('log_diameter'),
            # log velocity
            pl.col('relative_velocity_km/sec').log1p().alias('log_velocity'),
            # log distance
            pl.col('miss_distance_kilometers').log1p().alias('log_distance')
        ])

        return df
    
    def pipeline(self) -> pl.DataFrame:
        raw_data = self.collect_data()
        final_data_list = []
        for idx, data in enumerate(raw_data):
            formatted_data = self.format_data(data=data)
            feat_eng_data = self.feature_engineering(data=formatted_data)
            final_data_list.append(feat_eng_data)
            logger.info(f"Added formatted data to list: {idx+1}/{len(raw_data)}")
        final_df = pl.concat(final_data_list)
        return final_df