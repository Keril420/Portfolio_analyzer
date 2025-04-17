import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import logging



def create_stress_impact_heatmap(portfolio_data, stress_scenarios_results):
    """Creates a heat map of the impact of stress scenarios on different assets and sectors"""

    # Collecting data on sectors
    sectors = {}
    for asset in portfolio_data['assets']:
        if 'sector' in asset and asset['sector'] != 'N/A':
            sector = asset['sector']
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(asset['ticker'])

    # Create an Impact Matrix: Scenario x Asset
    asset_impact = {}
    for scenario_key, result in stress_scenarios_results.items():
        for ticker, impact in result['position_impacts'].items():
            if ticker not in asset_impact:
                asset_impact[ticker] = {}
            asset_impact[ticker][scenario_key] = impact['price_change'] * 100  # in percent

    # Create a DataFrame for the asset heatmap
    heatmap_data = []
    for ticker, impacts in asset_impact.items():
        row = {'Asset': ticker}
        row.update({scenario: impacts.get(scenario, 0) for scenario in stress_scenarios_results.keys()})
        heatmap_data.append(row)

    assets_df = pd.DataFrame(heatmap_data)

    # Create an Impact Matrix: Scenario x Sector
    sector_impact = {}
    for sector, tickers in sectors.items():
        sector_impact[sector] = {}
        for scenario in stress_scenarios_results.keys():
            # Calculate the average impact on the sector
            impacts = [asset_impact.get(ticker, {}).get(scenario, 0) for ticker in tickers]
            if impacts:
                sector_impact[sector][scenario] = sum(impacts) / len(impacts)
            else:
                sector_impact[sector][scenario] = 0

    # Create a DataFrame for the sector heatmap
    sector_data = []
    for sector, impacts in sector_impact.items():
        row = {'Sector': sector}
        row.update({scenario: impacts.get(scenario, 0) for scenario in stress_scenarios_results.keys()})
        sector_data.append(row)

    sectors_df = pd.DataFrame(sector_data)

    # Create a heat map for assets
    if not assets_df.empty and 'Asset' in assets_df.columns and len(assets_df.columns) > 1:
        fig_assets = go.Figure(data=go.Heatmap(
            z=assets_df.drop(columns=["Asset"]).values,
            x=assets_df.columns[1:],  # Scenario names
            y=assets_df["Asset"],
            colorscale="RdYlGn_r",
            colorbar=dict(title="Change (%)")
        ))
        fig_assets.update_layout(
            title="Impact of stress scenarios on assets (%)",
            xaxis_title="Scenario",
            yaxis_title="Asset",
        )

    else:
        # Create an empty image if there is not enough data
        fig_assets = go.Figure()
        fig_assets.update_layout(title='Not enough data for asset map')

    # Create a heat map for sectors
    if not sectors_df.empty and 'Sector' in sectors_df.columns and len(sectors_df.columns) > 1:
        fig_sectors = go.Figure(data=go.Heatmap(
            z=sectors_df.drop(columns=["Sector"]).values,
            x=sectors_df.columns[1:],
            y=sectors_df["Sector"],
            colorscale="RdYlGn_r",
            colorbar=dict(title="Change (%)")
        ))
        fig_sectors.update_layout(
            title="Impact of stress scenarios on sectors (%)",
            xaxis_title="Scenario",
            yaxis_title="Sector",
        )

    else:
        # Create an empty image if there is not enough data
        fig_sectors = go.Figure()
        fig_sectors.update_layout(title='Not enough data for sector map')

    return fig_assets, fig_sectors


def create_interactive_stress_impact_chart(stress_results, portfolio_value):
    """Creates an interactive chart with impact details per click"""

    # Data for the main chart
    scenarios = list(stress_results.keys())
    impacts = [result['portfolio_loss'] for result in stress_results.values()]
    percentages = [result['shock_percentage'] * 100 for result in stress_results.values()]


    fig = go.Figure()

    # Add the main diagram
    fig.add_trace(go.Bar(
        x=scenarios,
        y=impacts,
        name='Потери ($)',
        hovertemplate='<b>%{x}</b><br>Потери: $%{y:.2f}<br>Change: %{customdata:.2f}%',
        customdata=percentages,
        marker=dict(
            color=percentages,
            colorscale='RdYlGn_r',
            colorbar=dict(title='Change (%)')
        )
    ))

    # Add a horizontal line for 10% of the portfolio value
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=-portfolio_value * 0.1,
        x1=len(scenarios) - 0.5,
        y1=-portfolio_value * 0.1,
        line=dict(
            color="red",
            width=2,
            dash="dash",
        ),
        name="10% of the portfolio"
    )

    # Updating the layout
    fig.update_layout(
        title='Impact of stress scenarios on the portfolio',
        xaxis_title='Scenario',
        yaxis_title='Потери ($)',
        hovermode='closest',
        clickmode='event+select',
    )

    # Add annotation
    fig.add_annotation(
        x=0,
        y=-portfolio_value * 0.1,
        xref="x",
        yref="y",
        text="10% of the portfolio",
        showarrow=True,
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="#ffffff"
        ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#ff0000",
        ax=20,
        ay=-30,
        bordercolor="#ff0000",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ff0000",
        opacity=0.8
    )

    return fig


def create_risk_tree_visualization(portfolio_data, risk_factors=None):
    """Creates a hierarchical tree visualization of portfolio risk factors"""

    # Setting up logging for diagnostics
    logger = logging.getLogger('risk_tree')

    # Display diagnostic information
    if portfolio_data is None or 'assets' not in portfolio_data:
        logger.info("There is no portfolio or asset structure")
    else:
        logger.info(f"Portfolio with {len(portfolio_data['assets'])} assets received  ")

    # Check for correct portfolio data structure
    if portfolio_data is None or 'assets' not in portfolio_data or not portfolio_data['assets']:
        # Create a figure with an information message
        fig = go.Figure()
        fig.update_layout(
            title='Insufficient data to build a risk hierarchy',
            annotations=[dict(
                text='Add assets to your portfolio for risk analysis',
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.5
            )]
        )
        return fig


    data = []

    data.append(dict(
        id="Портфель",
        parent="",
        name="Портфель"
    ))

    # Adding the main risk categories
    risk_categories = ["Market risk", "Sector risk", "Specific risk"]
    for category in risk_categories:
        data.append(dict(
            id=category,
            parent="Портфель",
            name=category
        ))

    # Create groups for asset types
    asset_types = {"Stock": [], "Bonds": [], "Raw materials": [], "Currencies": []}
    sectors = {}

    # go through all assets
    total_weight = 0.0
    for asset in portfolio_data['assets']:
        weight = asset.get('weight', 0.0)
        total_weight += weight
        ticker = asset['ticker']

        # Determine the asset type (class)
        asset_type = "Stock"
        if 'asset_class' in asset:
            if asset['asset_class'] in ['Bond', 'Bonds', 'Fixed Income', 'Bonds']:
                asset_type = "Bonds"
            elif asset['asset_class'] in ['Commodity', 'Commodities', 'Raw materials']:
                asset_type = "Raw materials"
            elif asset['asset_class'] in ['Currency', 'Currencies', 'Currencies']:
                asset_type = "Currencies"

        # Add the asset to its type
        asset_types[asset_type].append((ticker, weight))

        # Add to specific risk
        data.append(dict(
            id=f"Specific risk|{ticker}",
            parent="Specific risk",
            name=ticker,
            value=weight * 100
        ))

        # Group by sector if there is information
        sector = asset.get('sector', 'Other')
        if sector == 'N/A':
            sector = 'Other'

        if sector not in sectors:
            sectors[sector] = []
            # Add a sector
            data.append(dict(
                id=f"Sector risk|{sector}",
                parent="Sector risk",
                name=sector
            ))

        sectors[sector].append((ticker, weight))

        # Add the asset to its sector
        data.append(dict(
            id=f"Sector risk|{sector}|{ticker}",
            parent=f"Sector risk|{sector}",
            name=ticker,
            value=weight * 100
        ))

    # Adding asset types to market risk
    for asset_type, assets in asset_types.items():
        if assets:  # If there are assets of this type
            # Add asset type
            data.append(dict(
                id=f"Market risk|{asset_type}",
                parent="Market risk",
                name=asset_type
            ))

            # Add assets of this type
            for ticker, weight in assets:
                data.append(dict(
                    id=f"Market risk|{asset_type}|{ticker}",
                    parent=f"Market risk|{asset_type}",
                    name=ticker,
                    value=weight * 100
                ))

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Create a sunburst chart
    try:
        fig = px.sunburst(
            df,
            ids='id',
            names='name',
            parents='parent',
            values='value',
            title='Hierarchy of portfolio risk factors',
            color_discrete_sequence=px.colors.qualitative.Set3
        )

        # Customize the appearance
        fig.update_layout(
            margin=dict(t=60, l=0, r=0, b=0),
            height=500
        )

    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")

        fig = go.Figure()
        fig.update_layout(
            title='Error creating risk hierarchy',
            annotations=[dict(
                text=f'Failed to create visualization: {str(e)}',
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.5
            )]
        )

    return fig