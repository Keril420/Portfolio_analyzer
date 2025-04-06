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
    """Создает иерархическую визуализацию факторов риска портфеля в виде дерева"""
    import plotly.graph_objects as go
    import pandas as pd
    import plotly.express as px
    import logging


    # Настраиваем логирование для диагностики
    logger = logging.getLogger('risk_tree')

    # Выводим диагностическую информацию
    if portfolio_data is None or 'assets' not in portfolio_data:
        logger.info("Отсутствует портфель или структура активов")
    else:
        logger.info(f"Получен портфель с {len(portfolio_data['assets'])} активами")

    # Проверяем наличие корректной структуры данных портфеля
    if portfolio_data is None or 'assets' not in portfolio_data or not portfolio_data['assets']:
        # Создаем фигуру с информационным сообщением
        fig = go.Figure()
        fig.update_layout(
            title='Недостаточно данных для построения иерархии рисков',
            annotations=[dict(
                text='Добавьте активы в портфель для анализа рисков',
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.5
            )]
        )
        return fig

    # Создаем более простую структуру данных для визуализации
    data = []

    # Добавляем корневой узел
    data.append(dict(
        id="Портфель",
        parent="",
        name="Портфель"
    ))

    # Добавляем основные категории риска
    risk_categories = ["Рыночный риск", "Секторный риск", "Специфический риск"]
    for category in risk_categories:
        data.append(dict(
            id=category,
            parent="Портфель",
            name=category
        ))

    # Создаем группы для типов активов
    asset_types = {"Акции": [], "Облигации": [], "Сырье": [], "Валюты": []}
    sectors = {}

    # Проходим по всем активам
    total_weight = 0.0
    for asset in portfolio_data['assets']:
        weight = asset.get('weight', 0.0)
        total_weight += weight
        ticker = asset['ticker']

        # Определяем тип актива (класс)
        asset_type = "Акции"  # По умолчанию считаем, что это акции
        if 'asset_class' in asset:
            if asset['asset_class'] in ['Bond', 'Bonds', 'Fixed Income', 'Облигации']:
                asset_type = "Облигации"
            elif asset['asset_class'] in ['Commodity', 'Commodities', 'Сырье']:
                asset_type = "Сырье"
            elif asset['asset_class'] in ['Currency', 'Currencies', 'Валюты']:
                asset_type = "Валюты"

        # Добавляем актив в его тип
        asset_types[asset_type].append((ticker, weight))

        # Добавляем в специфический риск
        data.append(dict(
            id=f"Специфический риск|{ticker}",
            parent="Специфический риск",
            name=ticker,
            value=weight * 100  # Умножаем на 100 для наглядности
        ))

        # Группируем по сектору, если есть информация
        sector = asset.get('sector', 'Прочее')
        if sector == 'N/A':
            sector = 'Прочее'

        if sector not in sectors:
            sectors[sector] = []
            # Добавляем сектор
            data.append(dict(
                id=f"Секторный риск|{sector}",
                parent="Секторный риск",
                name=sector
            ))

        sectors[sector].append((ticker, weight))

        # Добавляем актив в его сектор
        data.append(dict(
            id=f"Секторный риск|{sector}|{ticker}",
            parent=f"Секторный риск|{sector}",
            name=ticker,
            value=weight * 100
        ))

    # Добавляем типы активов в рыночный риск
    for asset_type, assets in asset_types.items():
        if assets:  # Если есть активы этого типа
            # Добавляем тип актива
            data.append(dict(
                id=f"Рыночный риск|{asset_type}",
                parent="Рыночный риск",
                name=asset_type
            ))

            # Добавляем активы этого типа
            for ticker, weight in assets:
                data.append(dict(
                    id=f"Рыночный риск|{asset_type}|{ticker}",
                    parent=f"Рыночный риск|{asset_type}",
                    name=ticker,
                    value=weight * 100
                ))

    # Создаем DataFrame
    df = pd.DataFrame(data)

    # Создаем sunburst-диаграмму
    try:
        fig = px.sunburst(
            df,
            ids='id',
            names='name',
            parents='parent',
            values='value',
            title='Иерархия факторов риска портфеля',
            color_discrete_sequence=px.colors.qualitative.Set3
        )

        # Настраиваем внешний вид
        fig.update_layout(
            margin=dict(t=60, l=0, r=0, b=0),
            height=500
        )

    except Exception as e:
        logger.error(f"Ошибка при создании визуализации: {str(e)}")
        # В случае ошибки создаем информационную фигуру
        fig = go.Figure()
        fig.update_layout(
            title='Ошибка при создании иерархии рисков',
            annotations=[dict(
                text=f'Не удалось создать визуализацию: {str(e)}',
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.5
            )]
        )

    return fig