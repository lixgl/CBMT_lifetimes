CBMT-lifetimes: Customer-Base Multi-Task Transformer for Cohort Analysis

A unified package for cohort-based customer lifetime value analysis using 
a multi-task Transformer model as described in the research paper 
"Multi-Task Learning for Customer Base Analysis".

Main Functions:
    - visualize_cohorts(): Visualize ROPC or AOV trends from raw transaction data
    - train_and_forecast(): Train CBMT model and generate forecasts
    - visualize_forecast(): Visualize historical + forecasted metrics

Usage:
    from CBMT_lifetimes import CohortLifetimes
    
    pipeline = CohortLifetimes()
    
    # Explore raw data
    pipeline.visualize_cohorts(df, metric='ropc')
    
    # Train & forecast
    results = pipeline.train_and_forecast(df, forecast_weeks=13)
    
    # Visualize results
    pipeline.visualize_forecast(results, metric='total_revenue')
"""
