import uuid
from flask import Flask, render_template, jsonify, request, session
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
import json
from datetime import datetime, timedelta, time
import uuid

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
game_states = {}
game_timestamps = {}


class TradingGame:
    def __init__(self, initial_capital=1000, shares_per_trade=1, time_step_minutes=10):
        self.initial_capital = initial_capital
        self.shares_per_trade = shares_per_trade  # Now using shares instead of dollar amount
        self.time_step_minutes = time_step_minutes
        # Shorting costs
        self.short_fee_daily_rate = 0.01  # 1% per day
        self.reset_game()


    def reset_game(self):
        self.capital = self.initial_capital
        self.position = 0  # Number of shares owned (negative for short)
        self.current_turn = 0
        self.price_history = [100.0]  # Start at $100
        self.volume_history = [5000]  # Add this line to initialize volume_history
        self.last_event_message = None  # Add this line to track extreme event messages

        # Start at 9:30 AM (market open)
        today = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
        self.timestamps = [today]

        self.trades = []  # List of (turn, action, price, quantity, pnl)
        self.portfolio_values = [self.initial_capital]
        self.short_fees_paid = 0
        self.market_closed = False

        # For OHLC data
        self.ohlc_data = [{
            'timestamp': today,
            'open': 100.0,
            'high': 100.0,
            'low': 100.0,
            'close': 100.0,
            'volume': 5000
        }]


    def to_dict(self):
        """Convert game state to JSON-serializable dictionary, with reduced size for session storage"""
        # Keep only the last 20 price/volume points to reduce session size
        max_history = 20
        if len(self.price_history) > max_history:
            price_history = self.price_history[-max_history:]
            volume_history = self.volume_history[-max_history:]
            timestamps = self.timestamps[-max_history:]
            ohlc_data = self.ohlc_data[-max_history:]
        else:
            price_history = self.price_history
            volume_history = self.volume_history
            timestamps = self.timestamps
            ohlc_data = self.ohlc_data

        return {
            'capital': self.capital,
            'position': self.position,
            'current_turn': self.current_turn,
            'price_history': price_history,
            'volume_history': volume_history,
            'timestamps': [ts.isoformat() for ts in timestamps],
            'trades': self.trades[-10:],  # Only keep the last 10 trades
            'portfolio_values': self.portfolio_values[-max_history:],
            'initial_capital': self.initial_capital,
            'increment': self.increment,
            'time_step_minutes': self.time_step_minutes,
            'short_fee_daily_rate': self.short_fee_daily_rate,
            'short_fees_paid': self.short_fees_paid,
            'market_closed': self.market_closed,
            'ohlc_data': [
                {
                    'timestamp': d['timestamp'].isoformat(),
                    'open': d['open'],
                    'high': d['high'],
                    'low': d['low'],
                    'close': d['close'],
                    'volume': d['volume']
                } for d in ohlc_data
            ]
        }

    def from_dict(self, data):
        """Load game state from dictionary"""
        self.capital = data['capital']
        self.position = data['position']
        self.current_turn = data['current_turn']
        self.price_history = data['price_history']
        self.volume_history = data.get('volume_history', [5000] * len(self.price_history))
        self.timestamps = [datetime.fromisoformat(ts) for ts in data['timestamps']]
        self.trades = data['trades']
        self.portfolio_values = data['portfolio_values']
        self.initial_capital = data['initial_capital']
        self.increment = data['increment']
        self.time_step_minutes = data['time_step_minutes']
        self.short_fee_daily_rate = data.get('short_fee_daily_rate', 0.01)
        self.short_fees_paid = data.get('short_fees_paid', 0)
        self.market_closed = data.get('market_closed', False)

        # Load OHLC data if available
        if 'ohlc_data' in data:
            self.ohlc_data = [
                {
                    'timestamp': datetime.fromisoformat(d['timestamp']),
                    'open': d['open'],
                    'high': d['high'],
                    'low': d['low'],
                    'close': d['close'],
                    'volume': d['volume']
                } for d in data['ohlc_data']
            ]
        else:
            # Create synthetic OHLC data if not available
            self.ohlc_data = []
            for i, price in enumerate(self.price_history):
                self.ohlc_data.append({
                    'timestamp': self.timestamps[i],
                    'open': price,
                    'high': price * 1.01,
                    'low': price * 0.99,
                    'close': price,
                    'volume': self.volume_history[i] if i < len(self.volume_history) else 5000
                })

    def generate_next_price(self):
        """Generate next price using a mixture model for fat tails"""
        current_price = self.price_history[-1]

        # Calculate probability of an extreme event
        # We want about 1.5 events per trading day (390 minutes)
        # Each time step is 10 minutes, so we have about 39 steps per day
        # Therefore probability is 1.5/39 ≈ 0.038 per step
        extreme_event_probability = 1.5 / 39

        # Decide if this is an extreme event
        if np.random.random() < extreme_event_probability:
            # Generate an extreme movement
            # Use a t-distribution with low degrees of freedom for fat tails
            t_dist = np.random.standard_t(df=3)  # 3 degrees of freedom gives fat tails

            # Scale the movement - larger than normal volatility
            # Multiply by 5-8x normal volatility
            volatility_multiplier = np.random.uniform(5, 12)
            base_volatility = 0.004  # Our normal volatility

            # Generate the change percent
            change_percent = (t_dist * base_volatility * volatility_multiplier)

            # Add a slight skew toward negative extreme events
            # (crashes tend to be more violent than rallies)
            if np.random.random() < 0.6:  # 60% of extreme events are negative
                change_percent = -abs(change_percent)

            # Add a message about what caused this extreme event
            event_messages = [
                "Breaking news: Major economic report surprises market!",
                "Sudden liquidity crisis hits the market!",
                "Unexpected Fed announcement rocks traders!",
                "Major hedge fund reported to be liquidating positions!",
                "Market reacts violently to geopolitical developments!",
                "Algorithmic trading causes flash movement in prices!",
                "Surprise earnings guidance from a competitor!",
                "Market rumors trigger massive position adjustments!",
                "Technical breakout causes cascade of orders!",
                "Large block trade creates temporary price dislocation!"
            ]

            # Store this message for display to the user
            self.last_event_message = np.random.choice(event_messages)
            print(f"Extreme event triggered: {change_percent:.2%} move")

        else:
            # Normal price movement with slight upward bias
            change_percent = np.random.normal(0.0006, 0.004)  # 0.01% mean, 0.4% std
            self.last_event_message = None

        # Apply the change to get new price
        new_price = current_price * (1 + change_percent)

        return max(new_price, 0.1), change_percent  # Return both price and percent change

    def calculate_short_fee(self, price, position, minutes):
        """Calculate short fee for the given position and time"""
        if position >= 0:  # No fee for long positions
            return 0

        # Convert daily rate to per-minute rate
        minute_rate = self.short_fee_daily_rate / (24 * 60)
        # Calculate fee based on absolute position value and time elapsed
        fee = abs(position) * price * minute_rate * minutes
        return fee

    def generate_realistic_ohlc(self, last_close, percent_change):
        """Generate realistic OHLC data with intra-bar volatility"""
        # Determine general direction and magnitude based on percent_change
        is_up_bar = percent_change > 0
        magnitude = abs(percent_change)

        # Generate a plausible open
        # The open is closer to the previous close with some gap potential
        gap_factor = np.random.normal(0, 0.003)  # Small random gap
        open_price = last_close * (1 + gap_factor)

        # Generate close based on the percent change from the original close
        close_price = last_close * (1 + percent_change)

        # Generate high and low with realistic intra-bar volatility
        # Volatility increases with magnitude of move
        base_volatility = 0.008  # Base volatility
        volatility = base_volatility + magnitude * 1.5  # Scaled by move size

        if is_up_bar:
            # Up bar - high extends above the higher of open/close
            high_extend = np.random.uniform(0.002, volatility)
            low_pullback = np.random.uniform(0.001, volatility * 0.7)

            high = max(open_price, close_price) * (1 + high_extend)
            low = min(open_price, close_price) * (1 - low_pullback)
        else:
            # Down bar - low extends below the lower of open/close
            low_extend = np.random.uniform(0.002, volatility)
            high_bounce = np.random.uniform(0.001, volatility * 0.7)

            low = min(open_price, close_price) * (1 - low_extend)
            high = max(open_price, close_price) * (1 + high_bounce)

        # Ensure high is highest, low is lowest
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)

        # Ensure reasonable bounds
        high = max(high, last_close * 0.99)
        low = min(low, last_close * 1.01)

        return open_price, high, low, close_price

    def generate_volume(self, percent_change):
        """Generate trading volume based on price change magnitude"""
        base_volume = 3000  # Base volume level

        # Check if this is likely an extreme event by looking at the magnitude of the change
        is_extreme_event = abs(percent_change) > 0.01  # 1% move is considered large

        if is_extreme_event:
            # Much higher volume on extreme events
            volatility_factor = 3 + abs(percent_change) * 50  # Even higher multiplier for extreme events
        else:
            # More volume on larger moves (absolute value of change)
            volatility_factor = 1 + abs(percent_change) * 25  # Amplify effect of volatility

        # Random component (lognormal to ensure positive values with occasional spikes)
        random_factor = np.random.lognormal(0, 0.5)  # Mean of 1, with variance

        # Add some time-of-day effects (higher volume at market open/close)
        hour = len(self.timestamps) % 39  # 39 10-minute periods in trading day
        if hour < 6 or hour > 33:  # First hour or last hour
            time_factor = 1.5
        else:
            time_factor = 1.0

        # Calculate volume
        volume = int(base_volume * volatility_factor * random_factor * time_factor)

        # Ensure some minimum and prevent extreme outliers
        return max(500, min(volume, 50000 if is_extreme_event else 25000))

    def execute_turn(self, action):
        """Execute a turn: 'buy', 'sell', 'short', or 'hold'"""
        # Check if market is closed - add early return here
        if self.market_closed:
            return {
                'price': self.price_history[-1],
                'capital': self.capital,
                'position': self.position,
                'portfolio_value': self.capital + (self.position * self.price_history[-1]),
                'trade': None,
                'market_closed': True,
                'message': 'Market is closed for the day. Please reset the game to start a new trading day.'
            }

        # Calculate new timestamp
        current_timestamp = self.timestamps[-1]
        new_timestamp = current_timestamp + timedelta(minutes=self.time_step_minutes)

        # Market closes at 4:00 PM sharp - check if we would exceed this
        market_open = self.timestamps[0]
        market_close = market_open.replace(hour=16, minute=0, second=0, microsecond=0)

        # Fix for off-by-one error: Check if current time is the last time step before market close
        if new_timestamp > market_close:
            print("Market should close!")
            self.market_closed = True
            current_price = self.price_history[-1]

            # Auto-liquidate any positions
            message = None
            if self.position != 0:
                if self.position > 0:
                    revenue = self.position * current_price
                    self.capital += revenue
                    message = f"Market closed at 4:00 PM! Auto-sold {self.position:.2f} shares at ${current_price:.2f}"
                else:
                    cost = abs(self.position) * current_price
                    self.capital -= cost
                    message = f"Market closed at 4:00 PM! Auto-covered {abs(self.position):.2f} shares at ${current_price:.2f}"

                self.position = 0
            else:
                message = "Market closed at 4:00 PM! Trading day complete."

            # Calculate final portfolio value
            current_value = self.capital

            return {
                'price': current_price,
                'capital': self.capital,
                'position': self.position,
                'portfolio_value': current_value,
                'trade': None,
                'market_closed': True,
                'message': message
            }

        # Market is still open, continue with normal turn processing
        # Generate new price
        last_close = self.price_history[-1]
        new_price, percent_change = self.generate_next_price()

        # Generate OHLC data with more realistic intrabar behavior
        open_price, high, low, close_price = self.generate_realistic_ohlc(last_close, percent_change)

        # Generate volume that correlates somewhat with price movement
        volume = self.generate_volume(percent_change)

        # Save the data - this is where we actually advance time
        self.price_history.append(close_price)
        self.volume_history.append(volume)
        self.timestamps.append(new_timestamp)
        self.current_turn += 1

        # Add OHLC data
        self.ohlc_data.append({
            'timestamp': new_timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })

        trade_info = None

        # Initialize message with any extreme event message
        message = self.last_event_message

        # Calculate short fees for existing short position if any
        if self.position < 0:
            short_fee = self.calculate_short_fee(close_price, self.position, self.time_step_minutes)
            self.capital -= short_fee
            self.short_fees_paid += short_fee
            # if message:
            #     message += f" | Short fee of ${short_fee:.2f} charged"
            # else:
            #     message = f"Short fee of ${short_fee:.2f} charged"

        # Calculate cost of shares at current price
        shares_value = self.shares_per_trade * close_price

        # Process trading actions - now using share-based logic instead of fixed dollar amounts
        if action == 'buy':
            # Check if user has enough capital to buy the shares
            if shares_value <= self.capital:
                # If short, buy means covering the short first
                if self.position < 0:
                    # Determine how many shares to cover (up to self.shares_per_trade)
                    shares_to_cover = min(abs(self.position), self.shares_per_trade)
                    cost = shares_to_cover * close_price

                    self.position += shares_to_cover
                    self.capital -= cost
                    trade_info = {
                        'turn': self.current_turn,
                        'action': 'cover',
                        'price': close_price,
                        'shares': shares_to_cover,
                        'cost': cost
                    }
                    self.trades.append(trade_info)
                    # if message:
                    #     message += f" | Covered {shares_to_cover:.2f} shares at ${close_price:.2f}"
                    # else:
                    #     message = f"Covered {shares_to_cover:.2f} shares at ${close_price:.2f}"
                else:
                    # Regular buy with exact share amount
                    self.position += self.shares_per_trade
                    self.capital -= shares_value
                    trade_info = {
                        'turn': self.current_turn,
                        'action': 'buy',
                        'price': close_price,
                        'shares': self.shares_per_trade,
                        'cost': shares_value
                    }
                    self.trades.append(trade_info)
                    # if message:
                    #     message += f" | Bought {self.shares_per_trade} shares at ${close_price:.2f}"
                    # else:
                    #     message = f"Bought {self.shares_per_trade} shares at ${close_price:.2f}"
            else:
                # Not enough capital
                if message:
                    message += f" | Insufficient funds to buy {self.shares_per_trade} shares at ${close_price:.2f}"
                else:
                    message = f"Insufficient funds to buy {self.shares_per_trade} shares at ${close_price:.2f}"

        elif action == 'sell':
            if self.position > 0:
                # Regular sell for long position
                shares_to_sell = min(self.shares_per_trade, self.position)
                revenue = shares_to_sell * close_price

                self.position -= shares_to_sell
                self.capital += revenue
                trade_info = {
                    'turn': self.current_turn,
                    'action': 'sell',
                    'price': close_price,
                    'shares': shares_to_sell,
                    'revenue': revenue
                }
                self.trades.append(trade_info)
                # if message:
                #     message += f" | Sold {shares_to_sell} shares at ${close_price:.2f}"
                # else:
                #     message = f"Sold {shares_to_sell} shares at ${close_price:.2f}"
            else:
                # Short sell or add to existing short
                # For shorting, we need enough capital to cover margin requirements
                # Let's use a simple 50% margin requirement
                margin_requirement = 0.5 * shares_value

                if margin_requirement <= self.capital:
                    self.position -= self.shares_per_trade  # Negative position means short
                    self.capital += shares_value  # Receive cash from short sale

                    trade_info = {
                        'turn': self.current_turn,
                        'action': 'short',
                        'price': close_price,
                        'shares': self.shares_per_trade,
                        'revenue': shares_value
                    }
                    self.trades.append(trade_info)
                    # if message:
                    #     message += f" | Shorted {self.shares_per_trade} shares at ${close_price:.2f}"
                    # else:
                    #     message = f"Shorted {self.shares_per_trade} shares at ${close_price:.2f}"
                else:
                    # Not enough capital for margin
                    if message:
                        message += f" | Insufficient funds for margin to short {self.shares_per_trade} shares at ${close_price:.2f}"
                    else:
                        message = f"Insufficient funds for margin to short {self.shares_per_trade} shares at ${close_price:.2f}"

        # Calculate current portfolio value
        current_value = self.capital + (self.position * close_price)
        self.portfolio_values.append(current_value)

        # Check if we're close to market close and warn the player
        time_to_close = (market_close - new_timestamp).total_seconds() / 60
        if time_to_close <= 20 and time_to_close > 10:
            if message:
                message += f" | Market closes in {int(time_to_close)} minutes!"
            else:
                message = f"Market closes in {int(time_to_close)} minutes!"

        return {
            'price': close_price,
            'capital': self.capital,
            'position': self.position,
            'portfolio_value': current_value,
            'trade': trade_info,
            'market_closed': self.market_closed,
            'message': message
        }

    def get_statistics(self):
        """Calculate trading statistics"""
        if len(self.portfolio_values) < 2:
            return {}

        portfolio_values = np.array(self.portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # P&L
        total_pnl = portfolio_values[-1] - self.initial_capital
        total_return = (portfolio_values[-1] / self.initial_capital - 1) * 100

        # Calculate buy and hold return - how the stock performed since market open
        initial_price = self.price_history[0]
        current_price = self.price_history[-1]
        buy_hold_return = ((current_price / initial_price) - 1) * 100

        # Hit rate (percentage of profitable trades)
        profitable_trades = 0
        total_trades = len([t for t in self.trades if t['action'] in ['buy', 'sell', 'short', 'cover']])

        # Simple hit rate calculation based on returns
        positive_returns = np.sum(returns > 0)
        hit_rate = (positive_returns / len(returns) * 100) if len(returns) > 0 else 0

        # Improved Sharpe ratio calculation
        if len(returns) > 1 and np.std(returns) > 0:
            # Mean return per period (10-minute periods in our game)
            mean_return = np.mean(returns)

            # Standard deviation of returns (volatility)
            std_return = np.std(returns, ddof=1)  # Use sample std deviation

            # More reasonable annualization factors:
            # - 39 periods per day (6.5 hours × 6 periods/hour) is correct
            # - But we need to be careful with annualization to avoid inflating the numbers

            # For annualization, we'll use a more conservative approach:
            # If we have less than a full day of data, scale to a full day only
            if self.current_turn < 39:  # Less than a full trading day
                # Scale to a daily figure
                periods_per_year = 252  # trading days per year
                daily_equivalent_mean = mean_return * (39 / max(1, self.current_turn))
                daily_equivalent_vol = std_return * np.sqrt(39 / max(1, self.current_turn))

                # Then annualize from daily
                annualized_return = daily_equivalent_mean * periods_per_year
                annualized_volatility = daily_equivalent_vol * np.sqrt(periods_per_year)
            else:
                # We have a full day or more of data, use standard annualization
                periods_per_year = 252  # trading days per year

                # Annualize by scaling to daily first
                daily_return = mean_return * 39  # 39 periods make a day
                daily_vol = std_return * np.sqrt(39)

                # Then annualize from daily
                annualized_return = daily_return * periods_per_year
                annualized_volatility = daily_vol * np.sqrt(periods_per_year)

            # Scale risk-free rate to the same period
            # Assuming 2% annual risk-free rate
            daily_risk_free = 0.02 / 252  # daily risk-free rate

            # Only use risk-adjusted Sharpe if we've completed a decent amount of trading
            if self.current_turn >= 5:
                try: print(f"{sharpe} = ({annualized_return} - 0.02) / {annualized_volatility}")
                except: pass
                sharpe = (annualized_return - 0.02) / annualized_volatility if annualized_volatility > 0 else 0
            else:
                # For very early in trading, don't subtract risk-free (to avoid extreme values)
                sharpe = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown) * 100

        # Market time info
        market_open = self.timestamps[0]
        current_time = self.timestamps[-1]
        market_close = market_open.replace(hour=16, minute=0, second=0)
        time_remaining = (market_close - current_time).total_seconds() // 60 if current_time < market_close else 0

        return {
            'total_pnl': round(total_pnl, 2),
            'total_return': round(total_return, 2),
            'buy_hold_return': round(buy_hold_return, 2),  # Added buy & hold return
            'hit_rate': round(hit_rate, 1),
            'sharpe_ratio': round(sharpe, 3),
            'max_drawdown': round(abs(max_drawdown), 2),
            'total_trades': total_trades,
            'current_price': round(self.price_history[-1], 2),
            'turns_played': self.current_turn,
            'short_fees_paid': round(self.short_fees_paid, 2),
            'current_time': current_time.strftime('%H:%M'),
            'market_close': market_close.strftime('%H:%M'),
            'time_remaining': round(time_remaining),
            'market_closed': self.market_closed
        }

    def get_chart_data(self):
        """Prepare data for Plotly candlestick chart with volume subplot"""
        if len(self.ohlc_data) < 1:
            # Fallback to line chart
            return {
                'x': [ts.strftime('%H:%M') for ts in self.timestamps],
                'y': self.price_history,
                'type': 'line'
            }

        return {
            'timestamps': [d['timestamp'].strftime('%H:%M') for d in self.ohlc_data],
            'open': [d['open'] for d in self.ohlc_data],
            'high': [d['high'] for d in self.ohlc_data],
            'low': [d['low'] for d in self.ohlc_data],
            'close': [d['close'] for d in self.ohlc_data],
            'volume': [d['volume'] for d in self.ohlc_data],
            'type': 'candlestick'
        }

# Initialize game
def get_game():
    """Get game instance - either from session or server storage"""
    # First, try to get game_id from session
    game_id = session.get('game_id')

    if not game_id or game_id not in game_states:
        # Create a new game
        game = TradingGame()
        game_id = str(uuid.uuid4())
        session['game_id'] = game_id
        game_states[game_id] = game
        return game
    else:
        # Return existing game from server-side storage
        return game_states[game_id]


# Modify save_game function
def save_game(game):
    """Save game state - server-side storage"""
    game_id = session.get('game_id')
    if game_id:
        game_states[game_id] = game
        game_timestamps[game_id] = datetime.now()


def cleanup_old_games():
    """Remove games that haven't been accessed in 30 minutes"""
    while True:
        time.sleep(600)  # Check every 10 minutes
        current_time = datetime.now()
        to_remove = []

        for game_id, timestamp in game_timestamps.items():
            # If game hasn't been accessed in 30 minutes
            if (current_time - timestamp).total_seconds() > 1800:
                to_remove.append(game_id)

        for game_id in to_remove:
            if game_id in game_states:
                del game_states[game_id]
            if game_id in game_timestamps:
                del game_timestamps[game_id]

        print(f"Cleaned up {len(to_remove)} inactive games. {len(game_states)} active games remain.")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/trade_history')
def trade_history():
    game = get_game()

    # Format trades for display in the UI
    formatted_trades = []
    for trade in game.trades:
        # Format timestamp
        timestamp = game.timestamps[trade['turn']-1].strftime('%H:%M') if trade['turn'] > 0 else game.timestamps[0].strftime('%H:%M')

        # Format action
        action = trade['action']
        action_display = action.capitalize()

        # Format price
        price = f"${trade['price']:.2f}"

        # Format shares
        shares = f"{trade['shares']:.2f}"

        # Format value (cost or revenue)
        if 'cost' in trade:
            value = f"-${trade['cost']:.2f}"
            type_class = 'negative'
        else:
            value = f"+${trade['revenue']:.2f}"
            type_class = 'positive'

        formatted_trades.append({
            'timestamp': timestamp,
            'action': action_display,
            'action_type': action,
            'price': price,
            'shares': shares,
            'value': value,
            'type_class': type_class
        })

    return jsonify({
        'trades': formatted_trades[::-1],  # Reverse to show newest first
        'position': round(game.position, 4),
        'position_value': round(game.position * game.price_history[-1], 2)
    })

@app.route('/api/game_state')
def game_state():
    game = get_game()
    current_price = game.price_history[-1] if game.price_history else 100.0
    portfolio_value = game.capital + (game.position * current_price)

    # Calculate the value of shares at current price
    shares_value = game.shares_per_trade * current_price

    # Calculate buy and hold return
    initial_price = game.price_history[0]
    buy_hold_return = ((current_price / initial_price) - 1) * 100

    # Get market time info
    market_open = game.timestamps[0]
    current_time = game.timestamps[-1]
    market_close = market_open.replace(hour=16, minute=0, second=0)
    time_remaining = (market_close - current_time).total_seconds() // 60 if current_time < market_close else 0

    return jsonify({
        'capital': round(game.capital, 2),
        'position': round(game.position, 4),
        'current_price': round(current_price, 2),
        'portfolio_value': round(portfolio_value, 2),
        'turn': game.current_turn,
        'buy_hold_return': round(buy_hold_return, 2),  # Added buy & hold return
        'shares_per_trade': game.shares_per_trade,
        'trade_value': round(shares_value, 2),
        'can_buy': game.capital >= shares_value,
        'can_sell': True,
        'short_fees_paid': round(game.short_fees_paid, 2),
        'is_short': game.position < 0,
        'current_time': current_time.strftime('%H:%M'),
        'market_close': market_close.strftime('%H:%M'),
        'time_remaining': round(time_remaining),
        'market_closed': game.market_closed
    })


@app.route('/api/execute_turn', methods=['POST'])
def execute_turn():
    game = get_game()
    action = request.json.get('action', 'hold')

    result = game.execute_turn(action)
    stats = game.get_statistics()
    chart_data = game.get_chart_data()

    # Save the updated game state
    save_game(game)

    # Calculate the value of shares at current price
    shares_value = game.shares_per_trade * result['price']

    # Calculate buy and hold return
    initial_price = game.price_history[0]
    current_price = result['price']
    buy_hold_return = ((current_price / initial_price) - 1) * 100

    return jsonify({
        'result': result,
        'stats': stats,
        'chart_data': chart_data,
        'game_state': {
            'capital': round(game.capital, 2),
            'position': round(game.position, 4),
            'current_price': round(result['price'], 2),
            'portfolio_value': round(result['portfolio_value'], 2),
            'turn': game.current_turn,
            'buy_hold_return': round(buy_hold_return, 2),  # Added buy & hold return
            'shares_per_trade': game.shares_per_trade,
            'trade_value': round(shares_value, 2),
            'can_buy': game.capital >= shares_value,
            'can_sell': True,
            'short_fees_paid': round(game.short_fees_paid, 2),
            'is_short': game.position < 0,
            'current_time': stats.get('current_time'),
            'market_close': stats.get('market_close'),
            'time_remaining': stats.get('time_remaining', 0),
            'market_closed': game.market_closed,
            'message': result.get('message')
        }
    })

@app.route('/api/reset_game', methods=['POST'])
def reset_game():
    game_id = session.get('game_id')
    if game_id and game_id in game_states:
        del game_states[game_id]

    # Create a new game
    game = TradingGame()
    game_id = str(uuid.uuid4())
    session['game_id'] = game_id
    game_states[game_id] = game

    return jsonify({'status': 'reset'})


@app.route('/api/chart_data')
def chart_data():
    game = get_game()
    return jsonify(game.get_chart_data())

if __name__ == '__main__':
    cleanup_thread = threading.Thread(target=cleanup_old_games, daemon=True)
    cleanup_thread.start()
    app.run(debug=True)
