# src/utils/calculations.py

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('calculations')


class PortfolioAnalytics:
    """
    Class for portfolio analytics and performance calculations.
    Provides methods for calculating returns, risk metrics, performance ratios, etc.
    """

    @staticmethod
    def calculate_returns(prices: pd.DataFrame, period: str = 'daily') -> pd.DataFrame:
        """
        Calculate returns for the specified period

        Args:
            prices: DataFrame with price data
            period: Period for returns calculation ('daily', 'weekly', 'monthly', 'annual')

        Returns:
            DataFrame with returns data
        """
        if prices.empty:
            return pd.DataFrame()

        if period == 'daily':
            return prices.pct_change().dropna()
        elif period == 'weekly':
            return prices.resample('W').last().pct_change().dropna()
        elif period == 'monthly':
            return prices.resample('M').last().pct_change().dropna()
        elif period == 'annual':
            return prices.resample('Y').last().pct_change().dropna()
        else:
            logger.warning(f"Unsupported period: {period}, using 'daily' as default")
            return prices.pct_change().dropna()

    @staticmethod
    def calculate_portfolio_return(returns: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
        """
        Calculate weighted portfolio returns

        Args:
            returns: DataFrame with returns for each asset
            weights: Dictionary with asset weights {ticker: weight}

        Returns:
            Series with portfolio returns
        """
        if returns.empty or not weights:
            return pd.Series()

        # Filter returns to include only assets in weights
        assets_in_weights = list(weights.keys())
        filtered_returns = returns[assets_in_weights].copy()

        # Calculate portfolio returns
        portfolio_weights = np.array([weights.get(ticker, 0) for ticker in filtered_returns.columns])
        portfolio_returns = filtered_returns.dot(portfolio_weights)

        return portfolio_returns

    @staticmethod
    def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
        """
        Calculate cumulative returns

        Args:
            returns: Series with returns data

        Returns:
            Series with cumulative returns
        """
        if returns.empty:
            return pd.Series()

        cumulative_returns = (1 + returns).cumprod() - 1
        return cumulative_returns

    @staticmethod
    def calculate_annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate annualized return

        Args:
            returns: Series with returns data
            periods_per_year: Number of periods in a year (252 for daily, 52 for weekly, 12 for monthly)

        Returns:
            Annualized return as a float
        """
        if returns.empty:
            return 0.0

        total_return = (1 + returns).prod() - 1
        years = len(returns) / periods_per_year

        if years <= 0:
            return 0.0

        annualized_return = (1 + total_return) ** (1 / years) - 1
        return annualized_return

    @staticmethod
    def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate annualized volatility

        Args:
            returns: Series with returns data
            periods_per_year: Number of periods in a year

        Returns:
            Annualized volatility as a float
        """
        if returns.empty:
            return 0.0

        volatility = returns.std() * np.sqrt(periods_per_year)
        return volatility

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe ratio

        Args:
            returns: Series with returns data
            risk_free_rate: Annual risk-free rate (e.g., 0.02 for 2%)
            periods_per_year: Number of periods in a year

        Returns:
            Sharpe ratio as a float
        """
        if returns.empty:
            return 0.0

        # Convert annual risk-free rate to period rate
        period_risk_free = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

        excess_returns = returns - period_risk_free
        annualized_excess_return = excess_returns.mean() * periods_per_year
        annualized_volatility = returns.std() * np.sqrt(periods_per_year)

        if annualized_volatility == 0:
            return 0.0

        sharpe_ratio = annualized_excess_return / annualized_volatility
        return sharpe_ratio

    # Исправляем проблему с делением на ноль в методе calculate_sortino_ratio
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Calculate Sortino ratio

        Args:
            returns: Series with returns data
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year

        Returns:
            Sortino ratio as a float
        """
        if returns.empty:
            return 0.0

        # Convert annual risk-free rate to period rate
        period_risk_free = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

        excess_returns = returns - period_risk_free
        annualized_excess_return = excess_returns.mean() * periods_per_year

        # Calculate downside deviation (only negative returns)
        negative_returns = returns[returns < period_risk_free] - period_risk_free

        if len(negative_returns) == 0 or abs(negative_returns.std()) < 1e-10:
            # Если нет отрицательной доходности, возвращаем большое значение (но не бесконечность)
            return 100.0  # Высокое значение вместо бесконечности

        # Используем стандартное отклонение отрицательных доходностей
        downside_deviation = np.sqrt(np.mean(negative_returns ** 2)) * np.sqrt(periods_per_year)

        if downside_deviation < 1e-10:  # Проверка на очень малое значение
            return 100.0  # Высокое значение вместо деления на малое число

        sortino_ratio = annualized_excess_return / downside_deviation
        return sortino_ratio

    # Исправляем метод расчета максимальной просадки
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """
        Calculate maximum drawdown

        Args:
            returns: Series with returns data

        Returns:
            Maximum drawdown as a float (positive value)
        """
        if returns.empty or len(returns) < 5:  # Требуем минимум 5 точек данных
            return 0.0

        try:
            cumulative_returns = (1 + returns).cumprod()
            peak_values = cumulative_returns.cummax()
            drawdowns = (cumulative_returns / peak_values) - 1
            max_drawdown = drawdowns.min()

            # Проверяем на выбросы (слишком большие просадки)
            if max_drawdown < -0.99:  # Просадка более 99% подозрительна
                logger.warning(f"Обнаружена экстремальная просадка: {max_drawdown:.2%}")
                # Можно применить винзоризацию или другую обработку выбросов

            # Return positive value for easier interpretation
            return abs(max_drawdown)
        except Exception as e:
            logger.error(f"Ошибка при расчете максимальной просадки: {e}")
            return 0.0

    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate Calmar ratio (annualized return / maximum drawdown)

        Args:
            returns: Series with returns data
            periods_per_year: Number of periods in a year

        Returns:
            Calmar ratio as a float
        """
        if returns.empty:
            return 0.0

        annualized_return = PortfolioAnalytics.calculate_annualized_return(returns, periods_per_year)
        max_drawdown = PortfolioAnalytics.calculate_max_drawdown(returns)

        if max_drawdown == 0:
            return 0.0

        calmar_ratio = annualized_return / max_drawdown
        return calmar_ratio

    @staticmethod
    def calculate_beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate beta (systematic risk) relative to a benchmark

        Args:
            returns: Series with portfolio returns
            benchmark_returns: Series with benchmark returns

        Returns:
            Beta as a float
        """
        if returns.empty or benchmark_returns.empty:
            return 0.0

        # Align the series
        common_index = returns.index.intersection(benchmark_returns.index)
        if len(common_index) == 0:
            return 0.0

        returns = returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]

        # Calculate beta using covariance and variance
        covariance = returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()

        if benchmark_variance == 0:
            return 0.0

        beta = covariance / benchmark_variance
        return beta

    @staticmethod
    def calculate_alpha(returns: pd.Series, benchmark_returns: pd.Series,
                        risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Calculate Jensen's alpha

        Args:
            returns: Series with portfolio returns
            benchmark_returns: Series with benchmark returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year

        Returns:
            Alpha as a float
        """
        if returns.empty or benchmark_returns.empty:
            return 0.0

        # Convert annual risk-free rate to period rate
        period_risk_free = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

        # Align the series
        common_index = returns.index.intersection(benchmark_returns.index)
        if len(common_index) == 0:
            return 0.0

        returns = returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]

        # Calculate beta
        beta = PortfolioAnalytics.calculate_beta(returns, benchmark_returns)

        # Calculate alpha (annualized)
        alpha = (returns.mean() - period_risk_free - beta * (
                benchmark_returns.mean() - period_risk_free)) * periods_per_year
        return alpha

    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) using historical method

        Args:
            returns: Series with returns data
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            VaR as a positive float (loss)
        """
        if returns.empty:
            return 0.0

        # VaR is a percentile of the returns distribution
        var = -np.percentile(returns, 100 * (1 - confidence_level))
        return var

    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall

        Args:
            returns: Series with returns data
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            CVaR as a positive float (expected loss)
        """
        if returns.empty:
            return 0.0

        # Calculate VaR
        var = PortfolioAnalytics.calculate_var(returns, confidence_level)

        # CVaR is the average of returns that are worse than VaR
        cvar = -returns[returns <= -var].mean()

        if np.isnan(cvar):
            return 0.0

        return cvar

    @staticmethod
    def calculate_portfolio_metrics(returns: pd.Series, benchmark_returns: pd.Series,
                                    risk_free_rate: float = 0.0, periods_per_year: int = 252) -> Dict:
        metrics = {}
        """
        Рассчитывает комплексные метрики производительности портфеля

        Args:
            returns: Серия доходностей портфеля
            benchmark_returns: Серия доходностей бенчмарка (опционально)
            risk_free_rate: Годовая безрисковая ставка
            periods_per_year: Количество периодов в году

        Returns:
            Словарь с метриками производительности
        """
        metrics = {}

        if returns.empty:
            return metrics

        # Базовые метрики доходности
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annualized_return'] = PortfolioAnalytics.calculate_annualized_return(returns, periods_per_year)
        metrics['volatility'] = PortfolioAnalytics.calculate_volatility(returns, periods_per_year)

        # Метрики риска
        metrics['sharpe_ratio'] = PortfolioAnalytics.calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
        metrics['sortino_ratio'] = PortfolioAnalytics.calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
        metrics['max_drawdown'] = PortfolioAnalytics.calculate_max_drawdown(returns)
        metrics['calmar_ratio'] = PortfolioAnalytics.calculate_calmar_ratio(returns, periods_per_year)

        # альфа, бета, информационный коэффициент
        if not benchmark_returns.empty:
            metrics['beta'] = PortfolioAnalytics.calculate_beta(returns, benchmark_returns)
        metrics['alpha'] = PortfolioAnalytics.calculate_alpha(returns, benchmark_returns,
                                                              risk_free_rate, periods_per_year)
        metrics['information_ratio'] = PortfolioAnalytics.calculate_information_ratio(returns, benchmark_returns)

        # Дополнительные метрики риска
        metrics['var_95'] = PortfolioAnalytics.calculate_var(returns, 0.95)
        metrics['cvar_95'] = PortfolioAnalytics.calculate_cvar(returns, 0.95)
        metrics['var_99'] = PortfolioAnalytics.calculate_var(returns, 0.99)
        metrics['cvar_99'] = PortfolioAnalytics.calculate_cvar(returns, 0.99)

        # Статистика распределения
        metrics['skewness'] = returns.skew() if len(returns) > 2 else 0
        metrics['kurtosis'] = returns.kurtosis() if len(returns) > 3 else 0

        # Метрики результативности
        win_metrics = PortfolioAnalytics.calculate_win_metrics(returns)
        metrics.update(win_metrics)

        # Метрики для разных временных периодов
        if isinstance(returns.index, pd.DatetimeIndex):
            period_performance = PortfolioAnalytics.calculate_period_performance(returns)

            # Добавляем с префиксом для разделения от других метрик
            for period, value in period_performance.items():
                metrics[f'period_{period}'] = value

        # Метрики относительно бенчмарка

        if benchmark_returns is not None:
            common_index = returns.index.intersection(benchmark_returns.index)

            if len(common_index) > 0:
                # Выравниваем серии по общему индексу
                aligned_returns = returns.loc[common_index]
                aligned_benchmark = benchmark_returns.loc[common_index]

                # Базовые метрики бенчмарка
                metrics['benchmark_return'] = (1 + aligned_benchmark).prod() - 1
                metrics['benchmark_annualized_return'] = PortfolioAnalytics.calculate_annualized_return(
                    aligned_benchmark, periods_per_year)
                metrics['benchmark_volatility'] = PortfolioAnalytics.calculate_volatility(
                    aligned_benchmark, periods_per_year)
                metrics['benchmark_max_drawdown'] = PortfolioAnalytics.calculate_max_drawdown(aligned_benchmark)

                # Коэффициенты бенчмарка
                metrics['benchmark_sharpe_ratio'] = PortfolioAnalytics.calculate_sharpe_ratio(
                    aligned_benchmark, risk_free_rate, periods_per_year)
                metrics['benchmark_sortino_ratio'] = PortfolioAnalytics.calculate_sortino_ratio(
                    aligned_benchmark, risk_free_rate, periods_per_year)
                metrics['benchmark_calmar_ratio'] = PortfolioAnalytics.calculate_calmar_ratio(
                    aligned_benchmark, periods_per_year)

                # Добавляем метрики результативности для бенчмарка
                benchmark_win_metrics = PortfolioAnalytics.calculate_win_metrics(aligned_benchmark)
                metrics.update({f'benchmark_{k}': v for k, v in benchmark_win_metrics.items()})

                # Дополнительные метрики риска бенчмарка
                metrics['benchmark_var_95'] = PortfolioAnalytics.calculate_var(aligned_benchmark, 0.95)
                metrics['benchmark_cvar_95'] = PortfolioAnalytics.calculate_cvar(aligned_benchmark, 0.95)

                # Capture Ratios
                capture_ratios = PortfolioAnalytics.calculate_capture_ratios(aligned_returns, aligned_benchmark)
                metrics.update({
                    'up_capture': capture_ratios['up_capture'],
                    'down_capture': capture_ratios['down_capture'],
                    'up_ratio': capture_ratios['up_ratio'],
                    'down_ratio': capture_ratios['down_ratio']
                })
        else:
            # Установить стандартные значения для бенчмарка, если он не предоставлен
            metrics['benchmark_total_return'] = 0.0
            metrics['benchmark_annualized_return'] = 0.0
            metrics['benchmark_volatility'] = 0.0
            metrics['benchmark_max_drawdown'] = 0.0
            metrics['benchmark_sharpe_ratio'] = 0.0
            metrics['benchmark_sortino_ratio'] = 0.0
            metrics['benchmark_calmar_ratio'] = 0.0
            metrics['benchmark_var_95'] = 0.0
            metrics['benchmark_cvar_95'] = 0.0
            metrics['beta'] = 1.0  # По умолчанию для бенчмарка
            metrics['alpha'] = 0.0
            metrics['information_ratio'] = 0.0
            metrics['up_capture'] = 1.0
            metrics['down_capture'] = 1.0

        return metrics

    @staticmethod
    def calculate_information_ratio(returns: pd.Series, benchmark_returns: pd.Series,
                                    periods_per_year: int = 252) -> float:
        """
        Рассчитывает Information Ratio - показатель риск-скорректированной доходности относительно бенчмарка
        """
        if returns.empty or benchmark_returns.empty:
            return 0.0

        # Выравниваем серии
        common_index = returns.index.intersection(benchmark_returns.index)
        returns = returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]

        # Рассчитываем избыточную доходность
        excess_returns = returns - benchmark_returns

        # Рассчитываем Information Ratio
        tracking_error = excess_returns.std() * np.sqrt(periods_per_year)
        if tracking_error == 0:
            return 0.0

        mean_excess = excess_returns.mean() * periods_per_year
        return mean_excess / tracking_error

    @staticmethod
    def calculate_capture_ratios(returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
        """
        Рассчитывает Up/Down Capture Ratios - насколько портфель захватывает
        восходящие/нисходящие движения бенчмарка
        """
        if returns.empty or benchmark_returns.empty:
            return {'up_capture': 0.0, 'down_capture': 0.0, 'up_ratio': 0.0, 'down_ratio': 0.0}

        # Выравниваем серии
        common_index = returns.index.intersection(benchmark_returns.index)
        returns = returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]

        # Разделяем на up и down периоды
        up_periods = benchmark_returns > 0
        down_periods = benchmark_returns < 0

        # Рассчитываем Up/Down Capture
        if up_periods.sum() > 0:
            up_capture = returns[up_periods].mean() / benchmark_returns[up_periods].mean()
        else:
            up_capture = 0.0

        if down_periods.sum() > 0:
            down_capture = returns[down_periods].mean() / benchmark_returns[down_periods].mean()
        else:
            down_capture = 0.0

        # Рассчитываем Up/Down Ratio
        up_ratio = (1 + returns[up_periods]).prod() / (
                    1 + benchmark_returns[up_periods]).prod() if up_periods.sum() > 0 else 1.0
        down_ratio = (1 + returns[down_periods]).prod() / (
                    1 + benchmark_returns[down_periods]).prod() if down_periods.sum() > 0 else 1.0

        return {
            'up_capture': up_capture,
            'down_capture': down_capture,
            'up_ratio': up_ratio,
            'down_ratio': down_ratio
        }

    @staticmethod
    def calculate_win_metrics(returns: pd.Series) -> Dict[str, float]:
        """
        Рассчитывает метрики результативности: Win Rate, Payoff Ratio и т.д.

        Args:
            returns: Серия доходностей

        Returns:
            Словарь с метриками результативности
        """
        if returns.empty or len(returns) < 2:
            return {
                'win_rate': 0.0,
                'payoff_ratio': 0.0,
                'profit_factor': 0.0
            }

        # Считаем положительные и отрицательные торговые дни
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        # Win Rate
        total_trades = len(wins) + len(losses)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0

        # Payoff Ratio (средний выигрыш / средний проигрыш)
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

        # Profit Factor (сумма выигрышей / сумма проигрышей)
        total_wins = wins.sum()
        total_losses = abs(losses.sum())
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        return {
            'win_rate': win_rate,  # В процентах
            'payoff_ratio': payoff_ratio,
            'profit_factor': profit_factor
        }

    @staticmethod
    def calculate_period_performance(returns: pd.Series, periods: Dict[str, Tuple[str, str]] = None) -> Dict[
        str, float]:
        """
        Рассчитывает доходность за различные периоды (MTD, YTD, 3Y, 5Y, и т.д.)

        Args:
            returns: Серия доходностей
            periods: Словарь периодов в формате {'имя_периода': (начальная_дата, конечная_дата)}
                    Если None, будут рассчитаны стандартные периоды

        Returns:
            Словарь с доходностями за указанные периоды
        """
        if returns.empty:
            return {}

        # Проверяем, что индекс - это DatetimeIndex
        if not isinstance(returns.index, pd.DatetimeIndex):
            return {}

        # Нормализуем индекс, чтобы устранить проблемы с часовыми поясами
        returns_index = returns.index.tz_localize(None) if returns.index.tz else returns.index
        returns = returns.copy()
        returns.index = returns_index

        result = {}
        today = returns_index.max()

        # Рассчитываем стандартные периоды, если не указаны пользовательские
        if periods is None:
            # Month to Date
            month_start = pd.Timestamp(today.year, today.month, 1)
            month_start_idx = returns_index[returns_index >= month_start]
            if len(month_start_idx) > 0:
                nearest_date = month_start_idx[0]
                result['MTD'] = (1 + returns[returns.index >= nearest_date]).prod() - 1

            # Year to Date
            year_start = pd.Timestamp(today.year, 1, 1)
            year_start_idx = returns_index[returns_index >= year_start]
            if len(year_start_idx) > 0:
                nearest_date = year_start_idx[0]
                result['YTD'] = (1 + returns[returns.index >= nearest_date]).prod() - 1

            # 1 Month
            one_month_ago = today - pd.DateOffset(months=1)
            one_month_idx = returns_index[returns_index >= one_month_ago]
            if len(one_month_idx) > 0:
                nearest_date = one_month_idx[0]
                result['1M'] = (1 + returns[returns.index >= nearest_date]).prod() - 1

            # 3 Months
            three_months_ago = today - pd.DateOffset(months=3)
            three_month_idx = returns_index[returns_index >= three_months_ago]
            if len(three_month_idx) > 0:
                nearest_date = three_month_idx[0]
                result['3M'] = (1 + returns[returns.index >= nearest_date]).prod() - 1

            # 6 Months
            six_months_ago = today - pd.DateOffset(months=6)
            six_month_idx = returns_index[returns_index >= six_months_ago]
            if len(six_month_idx) > 0:
                nearest_date = six_month_idx[0]
                result['6M'] = (1 + returns[returns.index >= nearest_date]).prod() - 1

            # 1 Year
            one_year_ago = today - pd.DateOffset(years=1)
            one_year_idx = returns_index[returns_index >= one_year_ago]
            if len(one_year_idx) > 0:
                nearest_date = one_year_idx[0]
                result['1Y'] = (1 + returns[returns.index >= nearest_date]).prod() - 1

            # 3 Years
            three_years_ago = today - pd.DateOffset(years=3)
            three_year_idx = returns_index[returns_index >= three_years_ago]
            if len(three_year_idx) > 0:
                nearest_date = three_year_idx[0]
                period_returns = returns[returns.index >= nearest_date]
                if len(period_returns) > 0:
                    period_total_return = (1 + period_returns).prod() - 1
                    # Если есть достаточно данных для годового расчета
                    if (today - nearest_date).days > 365:
                        years_fraction = (today - nearest_date).days / 365
                        result['3Y'] = (1 + period_total_return) ** (1 / years_fraction) - 1
                    else:
                        result['3Y'] = period_total_return

            # 5 Years
            five_years_ago = today - pd.DateOffset(years=5)
            five_year_idx = returns_index[returns_index >= five_years_ago]
            if len(five_year_idx) > 0:
                nearest_date = five_year_idx[0]
                period_returns = returns[returns.index >= nearest_date]
                if len(period_returns) > 0:
                    period_total_return = (1 + period_returns).prod() - 1
                    # Если есть достаточно данных для годового расчета
                    if (today - nearest_date).days > 365:
                        years_fraction = (today - nearest_date).days / 365
                        result['5Y'] = (1 + period_total_return) ** (1 / years_fraction) - 1
                    else:
                        result['5Y'] = period_total_return
        else:
            # Используем пользовательские периоды
            for period_name, (start_date, end_date) in periods.items():
                start_timestamp = pd.Timestamp(start_date).tz_localize(None)
                end_timestamp = pd.Timestamp(end_date).tz_localize(None)
                period_returns = returns[(returns.index >= start_timestamp) & (returns.index <= end_timestamp)]
                if not period_returns.empty:
                    result[period_name] = (1 + period_returns).prod() - 1

        return result