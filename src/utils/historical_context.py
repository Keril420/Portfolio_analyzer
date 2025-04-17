import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Database of historical crises
historical_crisis_context = {
    "financial_crisis_2008": {
        "name": "Financial crisis 2008-2009",
        "period": "September 2008 - March 2009",
        "trigger_events": [
            "Lehman Brothers Collapse September 15, 2008",
            "Subprime Mortgage Crisis",
            "Liquidity Problems in the Banking System"
        ],
        "key_indicators": [
            {"name": "TED Spread (the difference between LIBOR and Treasury Bills)", "value": "4.58%",
             "normal": "0.1-0.5%"},
            {"name": "VIX (volatility index)", "value": "80.86", "normal": "15-20"},
            {"name": "High Yield Bond Spread", "value": "21.82%", "normal": "3-5%"}
        ],
        "market_impact": {
            "S&P 500": "-56.8%",
            "MSCI World": "-54.0%",
            "US Real Estate": "-78.0%",
            "Oil": "-75.9%",
            "Gold": "+25.0%"
        },
        "policy_response": [
            "Fed Rate Cut to 0-0.25%",
            "Asset Purchase Program (QE1)",
            "Troubled Asset Relief Program (TARP)"
        ],
        "lessons_learned": [
            "The Importance of Controlling Financial Institutions' Leverage",
            "The Need for Transparency in Derivatives Markets",
            "Systemic Risks in the Financial System Require Systemic Oversight"
        ],
        "early_warning_signs": [
            "House price bubble",
            "Expansion of subprime lending",
            "Excessive securitization of mortgages",
            "High leverage in the financial system"
        ],
        "most_resilient_assets": [
            "US Treasuries",
            "Japanese Yen",
            "Gold",
            "Low Debt Companies with Stable Cash Flows"
        ],
        "most_affected_assets": [
            "Financials (especially banks and insurance companies)",
            "Real estate and real estate related companies",
            "Cyclic consumer goods",
            "Small companies with high debt levels"
        ]
    },
    "covid_2020": {
        "name": "COVID-19 Pandemic",
        "period": "February 2020 - March 2020",
        "trigger_events": [
            "Spread of coronavirus beyond China",
            "WHO declares pandemic on March 11, 2020",
            "Mass lockdowns and economic shutdowns"
        ],
        "key_indicators": [
            {"name": "VIX (volatility index)", "value": "82.69", "normal": "15-20"},
            {"name": "Economic Uncertainty Index", "value": "950", "normal": "100-150"},
            {"name": "Corporate Bond Spread", "value": "10.87%", "normal": "1-3%"}
        ],
        "market_impact": {
            "S&P 500": "-33.9%",
            "MSCI World": "-34.0%",
            "Oil": "-65.0%",
            "Gold": "-12.0% (temporarily, then growth)"
        },
        "policy_response": [
            "Fed Rate Cut to 0-0.25%",
            "Large-Scale Quantitative Easing Programs",
            "Financial Aid Packages (CARES Act in the US)",
            "Global Monetary and Fiscal Support"
        ],
        "lessons_learned": [
            "The Importance of Supply Chain Diversification",
            "The Need for Resilient Business Models and Strong Balance Sheets",
            "The Critical Role of Government Support in Systemic Crises",
            "Accelerating Digital Transformation"
        ],
        "early_warning_signs": [
            "Spread of the virus in China",
            "Disruption of supply chains",
            "Warnings from epidemiologists"
        ],
        "most_resilient_assets": [
            "Technology companies (especially those related to remote work)",
            "Pharmaceuticals and healthcare",
            "E-commerce companies",
            "US Treasuries"
        ],
        "most_affected_assets": [
            "Airlines and Travel Sector",
            "Traditional Retail",
            "Energy Sector",
            "Banks and Financial Institutions"
        ]
    },
    "tech_bubble_2000": {
        "name": "The Dot-com Crash",
        "period": "March 2000 - October 2002",
        "trigger_events": [
            "NASDAQ Peak March 10, 2000",
            "Microsoft Antitrust Case Ruling",
            "Revaluation of Internet Companies Without Revenue"
        ],
        "key_indicators": [
            {"name": "P/E NASDAQ", "value": ">200", "normal": "15-25"},
            {"name": "Number of IPOs of technology companies", "value": "record high", "normal": "moderate"},
            {"name": "Technology Share in the S&P 500", "value": "33%", "normal": "15-20%"}
        ],
        "market_impact": {
            "NASDAQ": "-78.0%",
            "S&P 500": "-49.1%",
            "Technology sector": "-83.0%"
        },
        "policy_response": [
            "Fed rate cut from 6.5% to 1.75%",
            "Fiscal stimulus (tax cuts))"
        ],
        "lessons_learned": [
            "The Dangers of Investing Based on Speculative Valuations",
            "The Importance of Sustainable Business Models and Real Returns",
            "The Risks of Sector Concentration"
        ],
        "early_warning_signs": [
            "Rapid growth in valuations without corresponding growth in earnings",
            "A sharp increase in the number of IPOs of loss-making companies",
            "Rapid growth in margin lending"
        ],
        "most_resilient_assets": [
            "Fixed Income",
            "Defensive Sectors (Healthcare, Utilities)",
            "Value Stocks",
            "Real Estate"
        ],
        "most_affected_assets": [
            "Internet and Dotcom Companies",
            "Telecommunications Sector",
            "Computer Equipment Manufacturers",
            "B2B Solution Providers"
        ]
    },
    "inflation_shock": {
        "name": "Inflation shock 2021-2022",
        "period": "End of 2021 - 2022",
        "trigger_events": [
            "Demand Recovery from the Pandemic",
            "Supply Chain Disruptions",
            "Energy Crisis",
            "Large Fiscal and Monetary Stimuli"
        ],
        "key_indicators": [
            {"name": "US Inflation (CPI)", "value": "9.1%", "normal": "2-3%"},
            {"name": "Energy prices", "value": "+76%", "normal": "±5-10%"},
            {"name": "Producer Price Index", "value": "11.3%", "normal": "1-3%"}
        ],
        "market_impact": {
            "S&P 500": "-20.0%",
            "Bonds (AGG)": "-17.0%",
            "NASDAQ": "-33.0%",
            "Gold": "-10.0%"
        },
        "policy_response": [
            "Higher Fed rates (2.25%-4.5% per year)",
            "Reduce Fed balance sheet (quantitative tightening)",
            "Similar actions by other central banks"
        ],
        "lessons_learned": [
            "Vulnerability of Global Supply Chains",
            "Risks of Simultaneous Fiscal and Monetary Stimuli",
            "The Importance of Preparing for Inflationary Periods",
            "Double Whammy for 60/40 Portfolios (Simultaneous Fall in Stocks and Bonds)"
        ],
        "early_warning_signs": [
            "Rising Commodity Prices",
            "Supply Chain Delays",
            "Record Money Supply (M2) Growth",
            "Record Low Interest Rates as Economy Recovers"
        ],
        "most_resilient_assets": [
            "Energy",
            "Commodities",
            "TIPS",
            "Value Stocks with Price Power"
        ],
        "most_affected_assets": [
            "Tech and growth stocks",
            "Long-term bonds",
            "Companies with low profitability or high energy costs",
            "Growth stocks with high multiples"
        ]
    }
}


def display_historical_context(scenario_key):
    """Displays detailed historical context for the selected scenario."""

    if scenario_key not in historical_crisis_context:
        st.warning(f"There is no historical context for the scenario. {scenario_key} ")
        return

    context = historical_crisis_context[scenario_key]

    st.subheader(f"Historical context: {context['name']}")

    # Basic information
    st.write(f"**Period:** {context['period']}")

    # Trigger Events
    st.subheader("Key events that triggered the crisis")
    for event in context['trigger_events']:
        st.markdown(f"- {event}")

    # Key indicators
    st.subheader("Key indicators")

    indicators_df = pd.DataFrame({
        'Indicator': [i['name'] for i in context['key_indicators']],
        'Importance in times of crisis': [i['value'] for i in context['key_indicators']],
        'Normal value': [i['normal'] for i in context['key_indicators']]
    })

    st.dataframe(indicators_df, use_container_width=True)

    # Impact on the market
    st.subheader("Impact on markets")

    impact_data = []
    for market, change in context['market_impact'].items():
        impact_data.append({
            'Market/Asset': market,
            'Change': change
        })

    impact_df = pd.DataFrame(impact_data)

    # Create an impact diagram if it is possible to convert changes into numbers
    try:
        # Trying to extract numeric values ​​from change lines
        numeric_changes = []
        for change in impact_df['Change']:
            # Remove the % and + signs, leaving only numbers and minus if any
            clean_change = change.replace('%', '').replace('+', '')
            # Separate the numeric part if there are brackets or text
            if '(' in clean_change:
                clean_change = clean_change.split('(')[0].strip()

            try:
                numeric_changes.append(float(clean_change))
            except ValueError:
                # If it can't be converted to a number, add 0
                numeric_changes.append(0)

        # Create a diagram
        fig_impact = px.bar(
            impact_df,
            x='Market/Asset',
            y=numeric_changes,
            color=numeric_changes,
            color_continuous_scale='RdYlGn',
            title='Impact on markets',
            labels={'y': 'Change (%)'}
        )

        fig_impact.update_traces(
            texttemplate='%{y}%',
            textposition='outside'
        )

        st.plotly_chart(fig_impact, use_container_width=True)

    except Exception as e:
        # If the diagram could not be created, just show the table
        st.warning(f"Failed to create diagram: {e}")
        st.dataframe(impact_df, use_container_width=True)

    # Two columns for other information
    col1, col2 = st.columns(2)

    with col1:
        # Countermeasures
        st.subheader("Regulatory responses")
        for policy in context['policy_response']:
            st.markdown(f"- {policy}")

        # The most sustainable assets
        st.subheader("The most sustainable assets")
        for asset in context['most_resilient_assets']:
            st.markdown(f"- {asset}")

    with col2:
        # Lessons Learned
        st.subheader("Lessons Learned")
        for lesson in context['lessons_learned']:
            st.markdown(f"- {lesson}")

        # The assets most affected
        st.subheader("The assets most affected")
        for asset in context['most_affected_assets']:
            st.markdown(f"- {asset}")

    # Early warning signs
    st.subheader("Early warning signs")
    for sign in context['early_warning_signs']:
        st.markdown(f"- {sign}")

    # Indicators that can predict such a crisis
    st.subheader("What to track today")

    st.info("""
    The following indicators may help identify similar conditions in the current economy:

      1. **Market Valuation**: Extreme P/E levels, especially in certain sectors
      2. **Credit Spreads**: Widening spreads can indicate rising stress
      3. **Volatility Index (VIX)**: Long periods of low volatility followed by sharp spikes
      4. **Leverage**: Large increases in corporate and consumer debt
      5. **Market Sentiment**: Excessive optimism or fear in investor sentiment
    """)

    # Modern parallels (if any)
    if 'current_parallels' in context:
        st.subheader("Modern parallels")
        for parallel in context['current_parallels']:
            st.markdown(f"- {parallel}")


def historical_analogy_page():
    """Page of historical analogies for the current state of the market"""

    st.title("Historical analogies")

    st.write("""
    On this page we compare the current market situation with historical periods 
    to identify possible analogies and learn lessons from the past..
    """)

    # Current market regime
    current_regime = st.selectbox(
        "Current market regime",
        [
            "Late Bull Market with High Valuations",
            "Beginning of a Bear Market",
            "Mid-Bear Market",
            "End of a Bear Market",
            "Beginning of a Bull Market",
            "Rising Inflation and Interest Rates",
            "Economic Recession",
            "Geopolitical Tensions"
        ]
    )

    # Determination of historical analogies depending on the selected mode
    historical_analogies = {
        "Late Phase Bull Market with High Valuations": [
            {"period": "1999-2000", "event": "Dotcom bubble", "similarity": 0.85},
            {"period": "2007", "event": "Before the financial crisis of 2008", "similarity": 0.75},
            {"period": "1972", "event": "Before the oil crisis", "similarity": 0.65}
        ],
        "The beginning of the bear market": [
            {"period": "Q4 2007", "event": "The beginning of the financial crisis", "similarity": 0.8},
            {"period": "Q1 2000", "event": "The beginning of the dot-com crash", "similarity": 0.7},
            {"period": "Q4 1972", "event": "Before the oil crisis", "similarity": 0.6}
        ],
        "The middle of a bear market": [
            {"period": "2008", "event": "Financial crisis", "similarity": 0.75},
            {"period": "2001", "event": "The Dot-com Crash", "similarity": 0.7},
            {"period": "1973-1974", "event": "Oil crisis", "similarity": 0.65}
        ],
        "The End of the Bear Market": [
            {"period": "Q1 2009", "event": "End of the financial crisis", "similarity": 0.8},
            {"period": "Q4 2002", "event": "The end of the dot-com bust", "similarity": 0.75},
            {"period": "1974", "event": "The end of the oil crisis", "similarity": 0.6}
        ],
        "The beginning of the bull market": [
            {"period": "2009-2010", "event": "After the financial crisis", "similarity": 0.85},
            {"period": "2003", "event": "After the dotcom crash", "similarity": 0.7},
            {"period": "1975", "event": "After the oil crisis", "similarity": 0.6}
        ],
        "Rising inflation and interest rates": [
            {"period": "1970-1980", "event": "A Decade of High Inflation", "similarity": 0.8},
            {"period": "1994", "event": "Fed Rate Hike", "similarity": 0.7},
            {"period": "2021-2022", "event": "Post-pandemic inflation", "similarity": 0.9}
        ],
        "Economic recession": [
            {"period": "2008-2009", "event": "The Great Recession", "similarity": 0.7},
            {"period": "2001-2002", "event": "Post-Dot-Com Recession", "similarity": 0.6},
            {"period": "1990-1991", "event": "Recession of the early 90s", "similarity": 0.5}
        ],
        "Geopolitical tensions": [
            {"period": "2022", "event": "Russia-Ukraine conflict", "similarity": 0.85},
            {"period": "2001", "event": "9/11 and the aftermath", "similarity": 0.6},
            {"period": "1990", "event": "Iraq's invasion of Kuwait", "similarity": 0.5}
        ]
    }

    if current_regime in historical_analogies:
        st.subheader(f"Historical analogies for the regime: {current_regime}")

        #We display analogies in the form of a table
        analogy_df = pd.DataFrame(historical_analogies[current_regime])
        analogy_df['similarity'] = analogy_df['similarity'] * 100

        # Visualization of the similarities of historical analogies
        fig_analogy = px.bar(
            analogy_df,
            x='period',
            y='similarity',
            color='similarity',
            color_continuous_scale='Viridis',
            text='event',
            title='Historical analogies and their similarities to the current situation',
            labels={'period': 'Historical period', 'similarity': 'Similarity (%)', 'event': 'Event'}
        )

        fig_analogy.update_traces(textposition='outside')
        fig_analogy.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')

        st.plotly_chart(fig_analogy, use_container_width=True)

        # Selecting a historical event for detailed study
        selected_period = st.selectbox(
            "Select a historical period for detailed study",
            options=analogy_df['period'].tolist(),
            format_func=lambda x: f"{x} - {analogy_df[analogy_df['period'] == x]['event'].values[0]}"
        )

        # Show a table of lessons learned
        historical_lessons = {
            "1970-1980": [
                "Inflation May Last Longer Than Expected",
                "Real Assets Outperform Financial Assets During High Inflation",
                "Small Rate Hike Rarely Stops Steady Inflation"
            ],
            "1990-1991": [
                "Geopolitical shocks can have short-term but strong market impacts," 
                "High-quality bonds and defensive stocks offer the best protection"
            ],
            "1994": [
                "A sudden rate hike could have a big impact on bond markets",
                "Short duration and floating rates help protect bond portfolios"
            ],
            "1999-2000": [
                "High valuations require high growth to justify",
                "Concentration in one sector greatly increases risk",
                "The importance of real earnings, not just revenue growth"
            ],
            "Q1 2000": [
                "Bubbles usually deflate faster than they inflate," 
                "The first wave of decline is often followed by a temporary rebound."
            ],
            "2001": [
                "Geopolitical shocks can accelerate already-started market trends",
                "Low valuations do not protect against short-term shocks"
            ],
            "2001-2002": [
                "Bear markets often have multiple waves," 
                "Fake rallies can mimic the start of a recovery"
            ],
            "Q4 2002": [
                "Market bottoms often form when pessimism is at its peak",
                "Leaders of the previous cycle rarely become leaders of the new cycle"
            ],
            "2003": [
                "New bull markets often start on bad economic news,"
                "Riskier assets tend to lead at the start of a new cycle."
            ],
            "2007": [
                "Low volatility often precedes major crises",
                "Credit spreads are often a better indicator than stock indices"
            ],
            "Q4 2007": [
                "Financial stocks often show signs of trouble first",
                "Widening credit spreads are an early indicator of trouble"
            ],
            "2008": [
                "Asset correlations tend to 1 during periods of severe crises",
                "Preserving liquidity is critical in severe crises"
            ],
            "2008-2009": [
                "Too much leverage can destroy even quality assets",
                "Economic recovery often lags market recovery"
            ],
            "Q1 2009": [
                "Market bottoms often occur before bad economic news ends,"
                "Extremely high volatility can signal a bottom is approaching"
            ],
            "2009-2010": [
                "Early stages of bull markets are usually strong, but with pullbacks,"
                "Lower quality, riskier stocks often lead the way at the start of a rally."
            ],
            "1990": [
                "Oil shocks could have cascading effects across the economy",
                "Defensive sectors offer relative safety"
            ],
            "1973-1974": [
                "The combination of geopolitical shocks and inflation is particularly damaging,"
                "Diversification beyond traditional asset classes is important"
            ],
            "1974": [
                "The best investment opportunities often come at times of maximum fear,"
                "Value stocks are usually better protected at the end of a bear market."
            ],
            "1975": [
                "Recovery could be swift and strong after deep sell-offs",
                "Inflationary assets could continue to perform even after inflation peaks"
            ],
            "1972": [
                "Low unemployment and high growth often precede inflation,"
                "Asset price bubbles can form even with moderate inflation"
            ],
            "2021-2022": [
                "Inflation Shock Could Hit Stocks and Bonds Together",
                "Value Stocks Could Outperform Growth Stocks During Rising Inflation",
                "Correlation Between Stocks and Bonds Could Turn Positive"
            ],
            "2022": [
                "Energy crises are particularly hard on energy importers",
                "Geopolitical risks are often underestimated by markets until they materialize",
                "Commodities can serve as a hedge against geopolitical risks"
            ]
        }

        if selected_period in historical_lessons:
            st.subheader(f"Rocks from the period {selected_period}")
            for lesson in historical_lessons[selected_period]:
                st.markdown(f"- {lesson}")

            # Recommendations for the current strategy
            st.subheader("Recommendations for the current investment strategy")

            recommendations = {
                "1970-1980": [
                    "Increase the share of real assets (commodities, real estate, TIPS)",
                    "Reduce the share of long-term bonds",
                    "Favor companies with strong pricing power"
                ],
                "1990-1991": [
                    "Increase allocation to high-quality bonds",
                    "Favor companies with low debt",
                    "Consider defensive sectors (utilities, consumer goods)"
                ],
                "1994": [
                    "Reduce bond duration",
                    "Consider floating rate bonds",
                    "Increase cash for future opportunities"
                ],
                "1999-2000": [
                    "Rebalance portfolio away from overheated sectors",
                    "Increase value stocks",
                    "Reduce concentration in tech sector"
                ],
                "Q1 2000": [
                    "Increase portfolio quality (companies with strong balance sheets)",
                    "Avoid false rebounds",
                    "Gradually increase positions in defensive sectors"
                ],
                "2001": [
                    "Increase cash allocation",
                    "Consider hedging strategies",
                    "Reduce exposure to cyclical sectors"
                ],
                "2001-2002": [
                    "Diversify your portfolio",
                    "Avoid companies with high debt",
                    "Focus on valuation levels, not momentum"
                ],
                "Q4 2002": [
                    "Start gradually increasing the share of stocks",
                    "Look for companies with strong models and low valuations",
                    "Consider small companies for long-term investments"
                ],
                "2003": [
                    "Increase the share of cyclical sectors",
                    "Consider international markets, especially emerging markets",
                    "Reduce the share of cash"
                ],
                "2007": [
                    "Reduce exposure to the financial sector",
                    "Increase the quality of credit instruments",
                    "Prepare for increased volatility"
                ],
                "Q4 2007": [
                    "Increase cash exposure",
                    "Reduce exposure to high-beta assets",
                    "Consider safe havens (gold, treasuries)"
                ],
                "2008": [
                    "Keep a significant portion of the portfolio in cash",
                    "Favor liquid instruments",
                    "Avoid using leverage"
                ],
                "2008-2009": [
                    "Focus on companies with low debt",
                    "Avoid sectors dependent on consumer credit",
                    "Gradually increase positions in quality stocks"
                ],
                "Q1 2009": [
                    "Start gradually increasing equity market exposure",
                    "Look for stocks with extremely low valuations",
                    "Consider highly rated corporate bonds"
                ],
                "2009-2010": [
                    "Increase cyclical sectors",
                    "Consider small caps",
                    "Reduce defensive assets"
                ],
                "1990": [
                    "Hedge energy risks",
                    "Increase the share of defensive sectors",
                    "Consider security-related assets"
                ],
                "1973-1974": [
                    "Increase the share of real assets",
                    "Reduce the share of bonds",
                    "Give preference to companies with stable cash flows"
                ],
                "1974": [
                    "Gradually increase exposure to stocks",
                    "Focus on value stocks",
                    "Look for companies with strong balance sheets"
                ],
                "1975": [
                    "Increase the share of cyclical sectors",
                    "Consider small companies",
                    "Keep part of the portfolio in real assets"
                ],
                "1972": [
                    "Prepare for rising inflation",
                    "Reduce exposure to high-multiplier growth stocks",
                    "Consider real assets as a hedge"
                ],
                "2021-2022": [
                    "Reduce long-term bonds",
                    "Increase value stocks",
                    "Consider inflation-protected assets (TIPS, commodities)"
                ],
                "2022": [
                    "Diversify energy sources",
                    "Consider companies in the defense sector",
                    "Increase the share of commodity assets"
                ]
            }

            if selected_period in recommendations:
                for recommendation in recommendations[selected_period]:
                    st.markdown(f"- {recommendation}")