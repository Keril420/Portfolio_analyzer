import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def create_stress_impact_heatmap(portfolio_data, stress_scenarios_results):
    """Создает тепловую карту влияния стресс-сценариев на различные активы и секторы"""

    # Собираем данные о секторах
    sectors = {}
    for asset in portfolio_data['assets']:
        if 'sector' in asset and asset['sector'] != 'N/A':
            sector = asset['sector']
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(asset['ticker'])

    # Создаем матрицу воздействия: сценарий x актив
    asset_impact = {}
    for scenario_key, result in stress_scenarios_results.items():
        for ticker, impact in result['position_impacts'].items():
            if ticker not in asset_impact:
                asset_impact[ticker] = {}
            asset_impact[ticker][scenario_key] = impact['price_change'] * 100  # в процентах

    # Создаем DataFrame для тепловой карты активов
    heatmap_data = []
    for ticker, impacts in asset_impact.items():
        row = {'Asset': ticker}
        row.update({scenario: impacts.get(scenario, 0) for scenario in stress_scenarios_results.keys()})
        heatmap_data.append(row)

    assets_df = pd.DataFrame(heatmap_data)

    # Создаем матрицу воздействия: сценарий x сектор
    sector_impact = {}
    for sector, tickers in sectors.items():
        sector_impact[sector] = {}
        for scenario in stress_scenarios_results.keys():
            # Рассчитываем среднее влияние на сектор
            impacts = [asset_impact.get(ticker, {}).get(scenario, 0) for ticker in tickers]
            if impacts:
                sector_impact[sector][scenario] = sum(impacts) / len(impacts)
            else:
                sector_impact[sector][scenario] = 0

    # Создаем DataFrame для тепловой карты секторов
    sector_data = []
    for sector, impacts in sector_impact.items():
        row = {'Sector': sector}
        row.update({scenario: impacts.get(scenario, 0) for scenario in stress_scenarios_results.keys()})
        sector_data.append(row)

    sectors_df = pd.DataFrame(sector_data)

    # Создаем тепловую карту для активов
    if not assets_df.empty and 'Asset' in assets_df.columns and len(assets_df.columns) > 1:
        fig_assets = px.imshow(
            assets_df.set_index('Asset').values,
            x=assets_df.columns[1:],  # Сценарии
            y=assets_df['Asset'],
            color_continuous_scale='RdYlGn_r',  # Обратная шкала: красный для негативных, зеленый для позитивных
            title='Влияние стресс-сценариев на активы (%)',
            labels={'x': 'Сценарий', 'y': 'Актив', 'color': 'Изменение (%)'}
        )
    else:
        # Создаем пустой рисунок, если данных недостаточно
        fig_assets = go.Figure()
        fig_assets.update_layout(title='Недостаточно данных для карты активов')

    # Создаем тепловую карту для секторов
    if not sectors_df.empty and 'Sector' in sectors_df.columns and len(sectors_df.columns) > 1:
        fig_sectors = px.imshow(
            sectors_df.set_index('Sector').values,
            x=sectors_df.columns[1:],  # Сценарии
            y=sectors_df['Sector'],
            color_continuous_scale='RdYlGn_r',
            title='Влияние стресс-сценариев на секторы (%)',
            labels={'x': 'Сценарий', 'y': 'Сектор', 'color': 'Изменение (%)'}
        )
    else:
        # Создаем пустой рисунок, если данных недостаточно
        fig_sectors = go.Figure()
        fig_sectors.update_layout(title='Недостаточно данных для карты секторов')

    return fig_assets, fig_sectors


def create_interactive_stress_impact_chart(stress_results, portfolio_value):
    """Создает интерактивную диаграмму с детализацией влияния по клику"""

    # Данные для основной диаграммы
    scenarios = list(stress_results.keys())
    impacts = [result['portfolio_loss'] for result in stress_results.values()]
    percentages = [result['shock_percentage'] * 100 for result in stress_results.values()]

    # Создаем фигуру
    fig = go.Figure()

    # Добавляем основную диаграмму
    fig.add_trace(go.Bar(
        x=scenarios,
        y=impacts,
        name='Потери ($)',
        hovertemplate='<b>%{x}</b><br>Потери: $%{y:.2f}<br>Изменение: %{customdata:.2f}%',
        customdata=percentages,
        marker=dict(
            color=percentages,
            colorscale='RdYlGn_r',
            colorbar=dict(title='Изменение (%)')
        )
    ))

    # Добавляем горизонтальную линию для 10% от стоимости портфеля
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
        name="10% портфеля"
    )

    # Обновляем макет
    fig.update_layout(
        title='Влияние стресс-сценариев на портфель',
        xaxis_title='Сценарий',
        yaxis_title='Потери ($)',
        hovermode='closest',
        clickmode='event+select',
    )

    # Добавляем аннотацию
    fig.add_annotation(
        x=0,
        y=-portfolio_value * 0.1,
        xref="x",
        yref="y",
        text="10% от портфеля",
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
    """Создает иерархическую визуализацию факторов риска в виде дерева"""

    # Если risk_factors не передан, создаем базовую структуру
    if risk_factors is None:
        risk_factors = {}

    # Организуем факторы риска по иерархии
    risk_hierarchy = {
        "Портфель": {
            "Рыночный риск": {
                "Акции": {},
                "Облигации": {},
                "Сырьё": {}
            },
            "Секторный риск": {},
            "Специфический риск": {},
            "Макроэкономический риск": {
                "Инфляция": {},
                "Процентные ставки": {},
                "Экономический рост": {}
            }
        }
    }

    # Заполняем структуру риска данными из портфеля
    sectors = {}
    for asset in portfolio_data['assets']:
        ticker = asset['ticker']
        weight = asset['weight']

        # Определяем тип актива
        asset_type = "Акции"  # По умолчанию
        if 'asset_class' in asset:
            if asset['asset_class'] in ['Bond', 'Bonds', 'Fixed Income']:
                asset_type = "Облигации"
            elif asset['asset_class'] in ['Commodity', 'Commodities']:
                asset_type = "Сырьё"

        # Добавляем в дерево
        if ticker not in risk_hierarchy["Портфель"]["Специфический риск"]:
            risk_hierarchy["Портфель"]["Специфический риск"][ticker] = {"weight": weight}

        # Группируем по секторам
        if 'sector' in asset and asset['sector'] != 'N/A':
            sector = asset['sector']
            if sector not in risk_hierarchy["Портфель"]["Секторный риск"]:
                risk_hierarchy["Портфель"]["Секторный риск"][sector] = {}

            if ticker not in risk_hierarchy["Портфель"]["Секторный риск"][sector]:
                risk_hierarchy["Портфель"]["Секторный риск"][sector][ticker] = {"weight": weight}

        # Группируем по типам активов
        if ticker not in risk_hierarchy["Портфель"]["Рыночный риск"][asset_type]:
            risk_hierarchy["Портфель"]["Рыночный риск"][asset_type][ticker] = {"weight": weight}

    # Преобразуем иерархическую структуру в формат для визуализации
    def build_sunburst_data(hierarchy, parent="", level=0):
        data = []

        for key, value in hierarchy.items():
            if key == "weight":
                continue

            # Для листьев дерева (активов)
            if isinstance(value, dict) and "weight" in value:
                data.append({
                    "id": key,
                    "parent": parent,
                    "value": value["weight"] * 100,
                    "level": level
                })
            # Для внутренних узлов дерева (категорий риска)
            else:
                # Формирование ID узла
                node_id = key if parent == "" else f"{parent}-{key}"

                data.append({
                    "id": node_id,
                    "parent": parent,
                    "value": None,  # Значение будет рассчитано автоматически
                    "level": level
                })

                # Рекурсивный обход дочерних узлов
                child_data = build_sunburst_data(value, node_id, level + 1)
                data.extend(child_data)

        return data

    # Создаем данные для визуализации
    sunburst_data = build_sunburst_data(risk_hierarchy)
    sunburst_df = pd.DataFrame(sunburst_data)

    # Создаем диаграмму иерархии рисков
    if not sunburst_df.empty and set(['id', 'parent', 'value']).issubset(sunburst_df.columns):
        fig = px.sunburst(
            sunburst_df,
            ids="id",
            parents="parent",
            values="value",
            color="level",
            color_continuous_scale="Blues",
            title="Иерархия факторов риска портфеля"
        )
    else:
        # Создаем пустой рисунок, если данных недостаточно
        fig = go.Figure()
        fig.update_layout(title='Недостаточно данных для дерева рисков')

    return fig