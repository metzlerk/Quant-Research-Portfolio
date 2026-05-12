import sys, os, traceback
sys.path.append(os.path.abspath('05_derivatives_pricing'))
try:
    from derivatives_pricing import OptionParams, BlackScholesPricer, put_call_parity
    params = OptionParams(spot=100, strike=100, maturity=1.0, rate=0.05, volatility=0.2, dividend_yield=0.01, option_type='call')
    call_price = BlackScholesPricer(params).price()
    put_params = OptionParams(spot=100, strike=100, maturity=1.0, rate=0.05, volatility=0.2, dividend_yield=0.01, option_type='put')
    put_price = BlackScholesPricer(put_params).price()
    residual = put_call_parity(params, call_price, put_price)
    print(f"call={call_price}, put={put_price}, residual={residual}")
except Exception:
    traceback.print_exc()
    raise
