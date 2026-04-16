import pandas as pd
import numpy as np
import os

def generate_alternative_credit_data(n_samples=5000):
    np.random.seed(42)
    
    # Feature 1: App Usage Intensity (High usage can correlate with tech-savviness)
    app_sessions = np.random.poisson(lam=10, size=n_samples)
    
    # Feature 2: SMS Financial Alerts (Proxies for income/transaction frequency)
    sms_alerts = np.random.poisson(lam=15, size=n_samples)
    
    # Feature 3: Utility Payment Punctuality (Standard alternative data point)
    # 0 = always on time, higher = more delays
    utility_delays = np.random.exponential(scale=2, size=n_samples).astype(int)
    
    # Feature 4: E-commerce Spending Ratio (Consumption behavior)
    ecommerce_spend = np.random.uniform(0, 0.6, size=n_samples)
    
    # Feature 5: Location Stability (GPS pings at same location overnight)
    location_stability = np.random.normal(loc=0.8, scale=0.1, size=n_samples)
    location_stability = np.clip(location_stability, 0, 1)
    
    # Feature 6: Device Age (Months)
    device_age = np.random.randint(1, 48, size=n_samples)
    
    # Target: Default Label (Simplified Logit-like probability)
    # Weights for risk factors
    z = (0.5 * utility_delays + 
         -0.8 * location_stability + 
         -0.1 * sms_alerts + 
         0.05 * device_age + 
         -2.0) # Base intercept
    
    prob = 1 / (1 + np.exp(-z))
    default_label = np.random.binomial(1, prob)
    
    df = pd.DataFrame({
        'user_id': range(1000, 1000 + n_samples),
        'app_sessions_per_day': app_sessions,
        'sms_fin_alerts_monthly': sms_alerts,
        'utility_delay_days': utility_delays,
        'ecommerce_spend_ratio': ecommerce_spend,
        'location_stability_score': location_stability,
        'device_age_months': device_age,
        'default_label': default_label
    })
    
    output_path = '/scratch/nishanth.r/pallavi/project_1_credit_scoring/alt_credit_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset generated at {output_path}")
    print(df.head())
    print("\nDefault Rate:", df['default_label'].mean())

if __name__ == "__main__":
    generate_alternative_credit_data()
