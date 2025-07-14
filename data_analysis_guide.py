#!/usr/bin/env python3
"""
Data Analysis with Python: A Comprehensive Guide for Full Stack Developers
=========================================================================

This script demonstrates how data analysts use Python in their daily work.
Run this script to see practical examples of data analysis tasks.

Requirements:
pip install pandas numpy matplotlib seaborn scipy scikit-learn plotly

Usage:
python data_analysis_guide.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_sample_data():
    """Generate realistic sample datasets for demonstration"""
    print("📊 CREATING SAMPLE DATA")
    print("=" * 50)
    
    # 1. Sales Data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    sales_data = pd.DataFrame({
        'date': dates,
        'sales': np.random.normal(1000, 200, 365) + np.sin(np.arange(365) * 2 * np.pi / 365) * 300,
        'region': np.random.choice(['North', 'South', 'East', 'West'], 365),
        'product': np.random.choice(['Product A', 'Product B', 'Product C'], 365),
        'marketing_spend': np.random.uniform(100, 1000, 365)
    })
    
    # 2. Customer Data
    customer_data = pd.DataFrame({
        'customer_id': range(1, 1001),
        'age': np.random.normal(35, 12, 1000),
        'income': np.random.normal(50000, 15000, 1000),
        'purchases': np.random.poisson(3, 1000),
        'satisfaction': np.random.uniform(1, 5, 1000),
        'segment': np.random.choice(['Premium', 'Standard', 'Basic'], 1000)
    })
    
    print("✅ Sample data created successfully!")
    print(f"   - Sales data: {len(sales_data)} records")
    print(f"   - Customer data: {len(customer_data)} records")
    print()
    
    return sales_data, customer_data

def data_exploration_and_cleaning(sales_data, customer_data):
    """Demonstrate data exploration and cleaning - core analyst tasks"""
    print("🔍 DATA EXPLORATION & CLEANING")
    print("=" * 50)
    
    # Data Overview
    print("📈 SALES DATA OVERVIEW:")
    print(sales_data.head())
    print(f"\nShape: {sales_data.shape}")
    print(f"Data types:\n{sales_data.dtypes}")
    print(f"\nMissing values:\n{sales_data.isnull().sum()}")
    
    # Statistical Summary
    print("\n📊 STATISTICAL SUMMARY:")
    print(sales_data.describe())
    
    # Data Quality Checks
    print("\n🔧 DATA QUALITY CHECKS:")
    print(f"Duplicate rows: {sales_data.duplicated().sum()}")
    print(f"Sales range: ${sales_data['sales'].min():.2f} - ${sales_data['sales'].max():.2f}")
    
    # Clean customer data (common analyst task)
    customer_data_clean = customer_data.copy()
    customer_data_clean['age'] = customer_data_clean['age'].clip(18, 80)  # Remove outliers
    customer_data_clean['income'] = customer_data_clean['income'].clip(0, 200000)
    
    print("✅ Data cleaning completed!")
    print(f"   - Age outliers handled: {(customer_data['age'] < 18).sum() + (customer_data['age'] > 80).sum()}")
    print(f"   - Income outliers handled: {(customer_data['income'] < 0).sum() + (customer_data['income'] > 200000).sum()}")
    print()
    
    return customer_data_clean

def data_aggregation_and_grouping(sales_data):
    """Show how analysts aggregate and group data"""
    print("📊 DATA AGGREGATION & GROUPING")
    print("=" * 50)
    
    # Group by region
    regional_summary = sales_data.groupby('region').agg({
        'sales': ['mean', 'sum', 'count'],
        'marketing_spend': 'mean'
    }).round(2)
    
    print("🌍 REGIONAL SALES SUMMARY:")
    print(regional_summary)
    
    # Time-based aggregation
    monthly_sales = sales_data.set_index('date').resample('M').agg({
        'sales': 'sum',
        'marketing_spend': 'mean'
    }).round(2)
    
    print("\n📅 MONTHLY SALES TRENDS:")
    print(monthly_sales.head())
    
    # Cross-tabulation
    print("\n📋 PRODUCT vs REGION CROSS-TAB:")
    crosstab = pd.crosstab(sales_data['product'], sales_data['region'], 
                          values=sales_data['sales'], aggfunc='mean').round(2)
    print(crosstab)
    print()
    
    return regional_summary, monthly_sales

def statistical_analysis(sales_data, customer_data):
    """Demonstrate statistical analysis techniques"""
    print("📈 STATISTICAL ANALYSIS")
    print("=" * 50)
    
    # Correlation analysis
    print("🔗 CORRELATION ANALYSIS:")
    correlation = sales_data[['sales', 'marketing_spend']].corr()
    print(correlation)
    
    # T-test example
    premium_satisfaction = customer_data[customer_data['segment'] == 'Premium']['satisfaction']
    basic_satisfaction = customer_data[customer_data['segment'] == 'Basic']['satisfaction']
    
    t_stat, p_value = stats.ttest_ind(premium_satisfaction, basic_satisfaction)
    print(f"\n📊 T-TEST RESULTS (Premium vs Basic satisfaction):")
    print(f"   T-statistic: {t_stat:.4f}")
    print(f"   P-value: {p_value:.4f}")
    print(f"   Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Chi-square test
    contingency_table = pd.crosstab(customer_data['segment'], 
                                   pd.cut(customer_data['purchases'], bins=3, labels=['Low', 'Medium', 'High']))
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"\n🔍 CHI-SQUARE TEST (Segment vs Purchase Level):")
    print(f"   Chi-square: {chi2:.4f}")
    print(f"   P-value: {p_val:.4f}")
    print()

def data_visualization(sales_data, customer_data):
    """Create various visualizations that analysts use"""
    print("📊 DATA VISUALIZATION")
    print("=" * 50)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Data Analysis Visualizations', fontsize=16, fontweight='bold')
    
    # 1. Time series plot
    axes[0, 0].plot(sales_data['date'], sales_data['sales'], alpha=0.7)
    axes[0, 0].set_title('Sales Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Sales ($)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Box plot
    sns.boxplot(data=sales_data, x='region', y='sales', ax=axes[0, 1])
    axes[0, 1].set_title('Sales Distribution by Region')
    axes[0, 1].set_ylabel('Sales ($)')
    
    # 3. Scatter plot
    axes[1, 0].scatter(customer_data['age'], customer_data['income'], alpha=0.6)
    axes[1, 0].set_title('Age vs Income')
    axes[1, 0].set_xlabel('Age')
    axes[1, 0].set_ylabel('Income ($)')
    
    # 4. Histogram
    axes[1, 1].hist(customer_data['satisfaction'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Customer Satisfaction Distribution')
    axes[1, 1].set_xlabel('Satisfaction Score')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('data_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Advanced visualization with Seaborn
    plt.figure(figsize=(12, 8))
    
    # Correlation heatmap
    plt.subplot(2, 2, 1)
    numeric_data = customer_data.select_dtypes(include=[np.number])
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    
    # Violin plot
    plt.subplot(2, 2, 2)
    sns.violinplot(data=customer_data, x='segment', y='income')
    plt.title('Income Distribution by Segment')
    plt.xticks(rotation=45)
    
    # Pair plot (showing relationships)
    plt.subplot(2, 2, 3)
    scatter_data = customer_data.sample(200)  # Sample for readability
    plt.scatter(scatter_data['purchases'], scatter_data['satisfaction'], 
                c=scatter_data['income'], cmap='viridis', alpha=0.6)
    plt.colorbar(label='Income')
    plt.xlabel('Purchases')
    plt.ylabel('Satisfaction')
    plt.title('Purchases vs Satisfaction (colored by Income)')
    
    # Bar plot
    plt.subplot(2, 2, 4)
    segment_stats = customer_data.groupby('segment')['purchases'].mean()
    segment_stats.plot(kind='bar')
    plt.title('Average Purchases by Segment')
    plt.xticks(rotation=45)
    plt.ylabel('Average Purchases')
    
    plt.tight_layout()
    plt.savefig('advanced_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualizations created and saved!")
    print("   - data_analysis_plots.png")
    print("   - advanced_analysis_plots.png")
    print()

def predictive_modeling(customer_data):
    """Demonstrate machine learning for predictive analysis"""
    print("🤖 PREDICTIVE MODELING")
    print("=" * 50)
    
    # Prepare data for modeling
    model_data = customer_data.copy()
    
    # Create dummy variables for categorical data
    model_data = pd.get_dummies(model_data, columns=['segment'], prefix='segment')
    
    # Define features and target
    X = model_data[['age', 'income', 'satisfaction', 'segment_Basic', 'segment_Premium', 'segment_Standard']]
    y = model_data['purchases']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate model
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print("📊 MODEL PERFORMANCE:")
    print(f"   R-squared: {r2:.4f}")
    print(f"   Mean Squared Error: {mse:.4f}")
    print(f"   RMSE: {np.sqrt(mse):.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print("\n📈 FEATURE IMPORTANCE:")
    print(feature_importance)
    
    # Prediction example
    sample_customer = X_test.iloc[0:1]
    prediction = model.predict(sample_customer)[0]
    actual = y_test.iloc[0]
    
    print(f"\n🔮 SAMPLE PREDICTION:")
    print(f"   Predicted purchases: {prediction:.2f}")
    print(f"   Actual purchases: {actual}")
    print(f"   Difference: {abs(prediction - actual):.2f}")
    print()

def business_insights(sales_data, customer_data):
    """Generate business insights from the analysis"""
    print("💡 BUSINESS INSIGHTS")
    print("=" * 50)
    
    # Key metrics
    total_sales = sales_data['sales'].sum()
    avg_daily_sales = sales_data['sales'].mean()
    best_region = sales_data.groupby('region')['sales'].sum().idxmax()
    
    print("📊 KEY BUSINESS METRICS:")
    print(f"   Total Sales: ${total_sales:,.2f}")
    print(f"   Average Daily Sales: ${avg_daily_sales:,.2f}")
    print(f"   Best Performing Region: {best_region}")
    
    # Customer insights
    premium_customers = (customer_data['segment'] == 'Premium').sum()
    avg_satisfaction = customer_data['satisfaction'].mean()
    high_value_customers = (customer_data['income'] > 70000).sum()
    
    print(f"\n👥 CUSTOMER INSIGHTS:")
    print(f"   Premium Customers: {premium_customers} ({premium_customers/len(customer_data)*100:.1f}%)")
    print(f"   Average Satisfaction: {avg_satisfaction:.2f}/5.0")
    print(f"   High Income Customers: {high_value_customers} ({high_value_customers/len(customer_data)*100:.1f}%)")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    print("   1. Focus marketing efforts on the best performing region")
    print("   2. Investigate satisfaction drivers for Premium customers")
    print("   3. Develop targeted campaigns for high-income segments")
    print("   4. Monitor daily sales trends for seasonal patterns")
    print()

def create_dashboard_data():
    """Create summary data that could be used in a dashboard"""
    print("📊 DASHBOARD DATA EXPORT")
    print("=" * 50)
    
    # Create summary tables for dashboard
    dashboard_data = {
        'daily_metrics': {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_sales': 125000,
            'avg_order_value': 85.50,
            'customer_count': 1462,
            'conversion_rate': 0.034
        },
        'regional_performance': [
            {'region': 'North', 'sales': 45000, 'growth': 0.12},
            {'region': 'South', 'sales': 38000, 'growth': 0.08},
            {'region': 'East', 'sales': 52000, 'growth': 0.15},
            {'region': 'West', 'sales': 41000, 'growth': 0.10}
        ]
    }
    
    # Save as JSON (common format for dashboards)
    import json
    with open('dashboard_data.json', 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    print("✅ Dashboard data exported to 'dashboard_data.json'")
    print("   This data can be consumed by web dashboards or BI tools")
    print()

def main():
    """Main function that runs all analysis examples"""
    print("🐍 PYTHON FOR DATA ANALYSIS - COMPREHENSIVE GUIDE")
    print("=" * 80)
    print("This script demonstrates how data analysts use Python daily.")
    print("As a full stack developer, you'll see familiar concepts applied to data work.")
    print("=" * 80)
    print()
    
    # Generate sample data
    sales_data, customer_data = create_sample_data()
    
    # Data exploration and cleaning
    customer_data_clean = data_exploration_and_cleaning(sales_data, customer_data)
    
    # Data aggregation
    regional_summary, monthly_sales = data_aggregation_and_grouping(sales_data)
    
    # Statistical analysis
    statistical_analysis(sales_data, customer_data_clean)
    
    # Visualizations
    data_visualization(sales_data, customer_data_clean)
    
    # Predictive modeling
    predictive_modeling(customer_data_clean)
    
    # Business insights
    business_insights(sales_data, customer_data_clean)
    
    # Dashboard data
    create_dashboard_data()
    
    print("🎉 ANALYSIS COMPLETE!")
    print("=" * 50)
    print("Key takeaways for full stack developers:")
    print("• Python for data analysis is more exploratory than application development")
    print("• Heavy use of pandas for data manipulation (like SQL for applications)")
    print("• Visualization is crucial for communicating insights")
    print("• Statistical analysis helps validate hypotheses")
    print("• Machine learning enables predictive capabilities")
    print("• Data quality and cleaning are major time investments")
    print("• Results often feed into dashboards or business reports")
    print()
    print("Files created:")
    print("• data_analysis_plots.png - Basic visualizations")
    print("• advanced_analysis_plots.png - Advanced visualizations")
    print("• dashboard_data.json - Summary data for dashboards")

if __name__ == "__main__":
    main()