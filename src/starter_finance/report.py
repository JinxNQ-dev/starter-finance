import click
import pandas as pd

from .metrics import (
    calculate_cagr,
    calculate_max_drawdown,
    calculate_sharpe,
    calculate_volatility,
)


@click.command()
@click.argument("csv_file")
def cli(csv_file):
    try:
        df = pd.read_csv(csv_file, parse_dates=["Date"], index_col="Date")
        returns = df["Portfolio"].pct_change().dropna()
        metrics = {
            "CAGR": calculate_cagr(df["Portfolio"]),
            "Volatility": calculate_volatility(returns),
            "Sharpe Ratio": calculate_sharpe(returns),
            "Max Drawdown": calculate_max_drawdown(df["Portfolio"]),
        }
        click.echo("Financial Report")
        for metric, value in metrics.items():
            click.echo(f"{metric}: {value:.4f}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


if __name__ == "__main__":
    cli()
