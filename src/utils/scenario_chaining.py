import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Определение структуры цепочек сценариев
scenario_chains = {
    "inflation_shock": {
        "name": "Инфляционный шок",
        "initial_impact": {"inflation": 5.0, "market": -0.05},
        "leads_to": [
            {
                "scenario": "rate_hike",
                "probability": 0.8,
                "delay": 30,  # дней
                "magnitude_modifier": 1.2  # усиление эффекта
            }
        ]
    },
    "rate_hike": {
        "name": "Повышение ставок",
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
        "name": "Кредитный кризис",
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
        "name": "Падение рынка недвижимости",
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
        "name": "Ослабление потребительского спроса",
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
        "name": "Экономическая рецессия",
        "initial_impact": {"market": -0.25, "unemployment": 3.0, "gdp": -2.0},
        "leads_to": []  # Конечное состояние
    }
}


def simulate_scenario_chain(starting_scenario, num_simulations=1000):
    """Симулирует возможные цепочки событий, начиная с заданного сценария"""
    all_results = []

    for _ in range(num_simulations):
        # Начинаем с исходного сценария
        current_scenario = starting_scenario
        chain = [current_scenario]
        total_impact = scenario_chains[current_scenario]["initial_impact"].copy()
        timeline = [0]  # дни с начала первого события

        # Продолжаем цепочку, пока есть последующие события
        while current_scenario in scenario_chains and "leads_to" in scenario_chains[current_scenario]:
            next_events = scenario_chains[current_scenario]["leads_to"]

            # Если нет последующих событий, завершаем цепочку
            if not next_events:
                break

            # Для каждого возможного последующего события
            triggered_next = False
            for next_event in next_events:
                # Определяем, произойдет ли событие (на основе вероятности)
                if random.random() < next_event["probability"]:
                    current_scenario = next_event["scenario"]
                    chain.append(current_scenario)
                    timeline.append(timeline[-1] + next_event["delay"])

                    # Суммируем влияние, с учетом модификатора силы
                    for factor, impact in scenario_chains[current_scenario]["initial_impact"].items():
                        if factor in total_impact:
                            total_impact[factor] += impact * next_event["magnitude_modifier"]
                        else:
                            total_impact[factor] = impact * next_event["magnitude_modifier"]

                    triggered_next = True
                    break

            # Если ни одно из последующих событий не сработало, завершаем цепочку
            if not triggered_next:
                break

        all_results.append({
            "chain": chain,
            "timeline": timeline,
            "total_impact": total_impact
        })

    return all_results


def visualize_scenario_chains(chain_results):
    """Создает санкей-диаграмму для визуализации цепочек сценариев"""
    # Подсчет переходов между сценариями
    transitions = {}
    for result in chain_results:
        chain = result["chain"]
        for i in range(len(chain) - 1):
            from_scenario = chain[i]
            to_scenario = chain[i + 1]
            key = (from_scenario, to_scenario)
            transitions[key] = transitions.get(key, 0) + 1

    # Создание данных для санкей-диаграммы
    source = []
    target = []
    value = []
    labels = []

    # Собираем уникальные сценарии
    all_scenarios = set()
    for from_s, to_s in transitions.keys():
        all_scenarios.add(from_s)
        all_scenarios.add(to_s)

    # Создаем индексы для сценариев
    scenario_indices = {scenario: i for i, scenario in enumerate(all_scenarios)}

    # Собираем данные для диаграммы
    for (from_s, to_s), count in transitions.items():
        source.append(scenario_indices[from_s])
        target.append(scenario_indices[to_s])
        value.append(count)

    labels = [scenario_chains[s]["name"] for s in all_scenarios]

    # Создаем санкей-диаграмму
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
    """Страница для анализа цепочек стрессовых событий"""
    st.title("Анализ цепочек стрессовых событий")

    st.write("""
    Стрессовые события редко происходят изолированно. Одно событие часто запускает цепочку 
    последующих событий, создавая сложные финансовые потрясения. Эта страница позволяет 
    моделировать такие цепочки и их совокупное влияние на рынки.
    """)

    starting_scenario = st.selectbox(
        "Выберите начальный сценарий",
        list(scenario_chains.keys()),
        format_func=lambda x: scenario_chains[x]["name"]
    )

    num_simulations = st.slider(
        "Количество симуляций",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100
    )

    if st.button("Моделировать цепочки событий"):
        with st.spinner("Выполнение моделирования..."):
            chain_results = simulate_scenario_chain(starting_scenario, num_simulations)

            # Визуализация цепочек
            st.plotly_chart(visualize_scenario_chains(chain_results), use_container_width=True)

            # Анализ совокупного воздействия
            impacts = {}
            for result in chain_results:
                for factor, value in result["total_impact"].items():
                    if factor not in impacts:
                        impacts[factor] = []
                    impacts[factor].append(value)

            # Статистика воздействия
            impact_stats = {}
            for factor, values in impacts.items():
                impact_stats[factor] = {
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "p5": np.percentile(values, 5),
                    "p95": np.percentile(values, 95)
                }

            # Отображение статистики
            impact_df = pd.DataFrame({
                "Фактор": list(impact_stats.keys()),
                "Среднее влияние": [stats["mean"] for stats in impact_stats.values()],
                "Медианное влияние": [stats["median"] for stats in impact_stats.values()],
                "5% квантиль": [stats["p5"] for stats in impact_stats.values()],
                "95% квантиль": [stats["p95"] for stats in impact_stats.values()]
            })

            st.subheader("Статистика совокупного воздействия")
            st.dataframe(impact_df.style.format({
                "Среднее влияние": "{:.2f}%",
                "Медианное влияние": "{:.2f}%",
                "5% квантиль": "{:.2f}%",
                "95% квантиль": "{:.2f}%"
            }), use_container_width=True)

            # Визуализация распределения воздействия
            st.subheader("Распределение совокупного воздействия")

            # Создаем боксплоты для каждого фактора
            fig = go.Figure()

            for factor, values in impacts.items():
                fig.add_trace(go.Box(
                    y=values,
                    name=factor,
                    boxmean=True  # добавляем среднее значение
                ))

            fig.update_layout(
                title="Распределение влияния по факторам",
                yaxis_title="Влияние (%)",
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

            # Отображение таблицы наиболее частых цепочек
            chain_counts = {}
            for result in chain_results:
                chain_str = " -> ".join([scenario_chains[s]["name"] for s in result["chain"]])
                chain_counts[chain_str] = chain_counts.get(chain_str, 0) + 1

            chain_df = pd.DataFrame({
                "Цепочка сценариев": list(chain_counts.keys()),
                "Частота появления": list(chain_counts.values()),
                "Частота (%)": [count / num_simulations * 100 for count in chain_counts.values()]
            }).sort_values("Частота появления", ascending=False).head(10)

            st.subheader("Топ-10 наиболее частых цепочек сценариев")
            st.dataframe(chain_df.style.format({
                "Частота (%)": "{:.2f}%"
            }), use_container_width=True)