def calculate_dcf(cash_flow_data, overview_data):
    # Simplified DCF Model
    try:
        if not cash_flow_data or not overview_data:
            return 0.0
        
        # Extract latest annual report
        reports = cash_flow_data.get('annualReports', [])
        if not reports:
            return 0.0
            
        latest_report = reports[0]
        op_cash = float(latest_report.get('operatingCashflow', 0))
        capex = float(latest_report.get('capitalExpenditures', 0))
        
        # Free Cash Flow (FCF)
        fcf = op_cash - capex
        
        # Assumptions (Can be made dynamic in future versions)
        growth_rate = 0.05  # 5% conservative growth
        discount_rate = 0.10 # 10% WACC
        terminal_growth = 0.025
        shares_outstanding = float(overview_data.get('SharesOutstanding', 1))
        
        # 5 Year Projection
        future_fcf = []
        projected_fcf = fcf
        for i in range(1, 6):
            projected_fcf = projected_fcf * (1 + growth_rate)
            discounted_fcf = projected_fcf / ((1 + discount_rate) ** i)
            future_fcf.append(discounted_fcf)
            
        # Terminal Value
        terminal_val = (projected_fcf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
        terminal_val_discounted = terminal_val / ((1 + discount_rate) ** 5)
        
        total_value = sum(future_fcf) + terminal_val_discounted
        intrinsic_value = total_value / shares_outstanding
        
        return round(intrinsic_value, 2)
    except Exception as e:
        print(f"DCF Error: {e}")
        return 0.0
