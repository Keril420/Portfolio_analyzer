import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Defining the structure of script chains
scenario_chains = {
    "inflation_shock": {
        "name": "Inflation shock",
        "initial_impact": {"inflation": 5.0, "market": -0.05},
        "leads_to": [
            {
                "scenario": "rate_hike",
                "probability": 0.8,
                "delay": 30,  # days
                "magnitude_modifier": 1.2  # enhance effect
            }
        ]
    },
    "rate_hike": {
        "name": "Raising the stakes",
        "initial_impact": {"interest_rates": 1.5, "bonds": -0.10, "tech_stocks": -0.15},
        "leads_to": [
            {
                "scenario": "credit_crunch",
                "probability": 0.6,
                "delay": 60,
                "magnitude_modifier": 1.0
            },
            {
                "scenario": "housing_decline",
                "probability": 0.7,
                "delay": 90,
                "magnitude_modifier": 0.9
            }
        ]
    },
    "credit_crunch": {
        "name": "Credit Crisis",
        "initial_impact": {"financials": -0.20, "consumer_discretionary": -0.15, "market": -0.10},
        "leads_to": [
            {
                "scenario": "recession",
                "probability": 0.7,
                "delay": 120,
                "magnitude_modifier": 1.1
            }
        ]
    },
    "housing_decline": {
        "name": "The real estate market is falling",
        "initial_impact": {"real_estate": -0.25, "financials": -0.10, "construction": -0.20},
        "leads_to": [
            {
                "scenario": "consumer_weakness",
                "probability": 0.65,
                "delay": 60,
                "magnitude_modifier": 0.9
            }
        ]
    },
    "consumer_weakness": {
        "name": "Weakening consumer demand",
        "initial_impact": {"consumer_discretionary": -0.15, "retail": -0.20, "market": -0.05},
        "leads_to": [
            {
                "scenario": "recession",
                "probability": 0.6,
                "delay": 90,
                "magnitude_modifier": 1.0
            }
        ]
    },
    "recession": {
        "name": "Economic recession",
        "initial_impact": {"market": -0.25, "unemployment": 3.0, "gdp": -2.0},
        "leads_to": []  # Final state
    }
}


def simulate_scenario_chain(starting_scenario, num_simulations=1000):
    """Simulates possible chains of events starting from a given scenario"""
    all_results = []

    for _ in range(num_simulations):
        # Let's start with the original script
        current_scenario = starting_scenario
        chain = [current_scenario]
        total_impact = scenario_chains[current_scenario]["initial_impact"].copy()
        timeline = [0]  # days since the start of the first event

        # continue the chain as long as there are subsequent events
        while current_scenario in scenario_chains and "leads_to" in scenario_chains[current_scenario]:
            next_events = scenario_chains[current_scenario]["leads_to"]

            # If there are no subsequent events, end the chain
            if not next_events:
                break

            # For each possible subsequent event
            triggered_next = False
            for next_event in next_events:
                # Determine whether an event will occur (based on probability)
                if random.random() < next_event["probability"]:
                    current_scenario = next_event["scenario"]
                    chain.append(current_scenario)
                    timeline.append(timeline[-1] + next_event["delay"])

                    # Sum up the influence, taking into account the strength modifier
                    for factor, impact in scenario_chains[current_scenario]["initial_impact"].items():
                        if factor in total_impact:
                            total_impact[factor] += impact * next_event["magnitude_modifier"]
                        else:
                            total_impact[factor] = impact * next_event["magnitude_modifier"]

                    triggered_next = True
                    break

            # If none of the subsequent events triggered, terminate the chain
            if not triggered_next:
                break

        all_results.append({
            "chain": chain,
            "timeline": timeline,
            "total_impact": total_impact
        })

    return all_results


def visualize_scenario_chains(chain_results):
    """Creates a sankey diagram to visualize scenario chains"""
    # Counting transitions between scenarios
    transitions = {}
    for result in chain_results:
        chain = result["chain"]
        for i in range(len(chain) - 1):
            from_scenario = chain[i]
            to_scenario = chain[i + 1]
            key = (from_scenario, to_scenario)
            transitions[key] = transitions.get(key, 0) + 1

    # Creating data for a sankey chart
    source = []
    target = []
    value = []
    labels = []

    # We collect unique scenarios
    all_scenarios = set()
    for from_s, to_s in transitions.keys():
        all_scenarios.add(from_s)
        all_scenarios.add(to_s)

    # We collect unique scenarios
    scenario_indices = {scenario: i for i, scenario in enumerate(all_scenarios)}

    # Collecting data for the chart
    for (from_s, to_s), count in transitions.items():
        source.append(scenario_indices[from_s])
        target.append(scenario_indices[to_s])
        value.append(count)

    labels = [scenario_chains[s]["name"] for s in all_scenarios]

    # Create a sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color="blue"
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        )
    )])

    fig.update_layout(title_text="Цепочки сценариев", font_size=10)
    return fig


def scenario_chaining_page():
    """Stress Event Chain Analysis Page"""
    st.title("Stress Event Chain Analysis")

    st.write("""
    Stress events rarely occur in isolation. 
    A single event often triggers a chain of subsequent events, 
    creating complex financial shocks. 
    This page allows you to model such chains 
    and their combined impact on markets.
    """)

    starting_scenario = st.selectbox(
        "Select the initial scenario",
        list(scenario_chains.keys()),
        format_func=lambda x: scenario_chains[x]["name"]
    )

    num_simulations = st.slider(
        "Number of simulations",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100
    )

    if st.button("Model chains of events"):
        with st.spinner("Performing the simulation..."):
            chain_results = simulate_scenario_chain(starting_scenario, num_simulations)

            # Visualization of chains
            st.plotly_chart(visualize_scenario_chains(chain_results), use_container_width=True)

            # Total Impact Analysis
            impacts = {}
            for result in chain_results:
                for factor, value in result["total_impact"].items():
                    if factor not in impacts:
                        impacts[factor] = []
                    impacts[factor].append(value)

            # Impact Statistics
            impact_stats = {}
            for factor, values in impacts.items():
                impact_stats[factor] = {
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "p5": np.percentile(values, 5),
                    "p95": np.percentile(values, 95)
                }

            # Display statistics
            impact_df = pd.DataFrame({
                "Factor": list(impact_stats.keys()),
                "Average influence": [stats["mean"] for stats in impact_stats.values()],
                "Median influence": [stats["median"] for stats in impact_stats.values()],
                "5% quantile": [stats["p5"] for stats in impact_stats.values()],
                "95% quantile": [stats["p95"] for stats in impact_stats.values()]
            })

            st.subheader("Total Impact Statistics")
            st.dataframe(impact_df.style.format({
                "Average influence": "{:.2f}%",
                "Median influence": "{:.2f}%",
                "5% quantile": "{:.2f}%",
                "95% quantile": "{:.2f}%"
            }), use_container_width=True)

            # Visualization of impact distribution
            st.subheader("Distribution of the total impact")

            # Create boxplots for each factor
            fig = go.Figure()

            for factor, values in impacts.items():
                fig.add_trace(go.Box(
                    y=values,
                    name=factor,
                    boxmean=True  # add the average value
                ))

            fig.update_layout(
                title="Distribution of influence by factors",
                yaxis_title="Influence (%)",
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

            # Display the table of the most frequent chains
            chain_counts = {}
            for result in chain_results:
                chain_str = " -> ".join([scenario_chains[s]["name"] for s in result["chain"]])
                chain_counts[chain_str] = chain_counts.get(chain_str, 0) + 1

            chain_df = pd.DataFrame({
                "Scenario chain": list(chain_counts.keys()),
                "Frequency of occurrence": list(chain_counts.values()),
                "Frequency (%)": [count / num_simulations * 100 for count in chain_counts.values()]
            }).sort_values("Frequency of occurrence", ascending=False).head(10)

            st.subheader("Top 10 Most Common Scenario Chains")
            st.dataframe(chain_df.style.format({
                "Frequency (%)": "{:.2f}%"
            }), use_container_width=True)