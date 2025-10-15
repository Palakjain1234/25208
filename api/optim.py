"""
optim.py

Optimization engine for Rake Formation System.
Uses models from multi_model.py to:
- Predict costs and probabilities.
- Perform MILP optimization for rake assignment.
"""

import pandas as pd
import numpy as np
import os
import time
from pulp import (
    LpVariable, LpProblem, LpMinimize, lpSum, PULP_CBC_CMD, LpStatus
)
from multi_model import MultiModelPipeline
import joblib

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
DATA_PATH = r"C:\Installation\Hackathons\sih208\bokaro_to_cmo_customers.csv" 
MODEL_DIR = r"C:\Installation\Hackathons\sih208\saved_models"
OUTPUT_PATH = r"C:\Installation\Hackathons\sih208\optimized_rake_allocations.csv"

ON_TIME_THRESHOLD = 0.6
PRIORITY_WEIGHT = 0.05
RELIABILITY_WEIGHT = 0.15  # Increased further
MIN_RAKE_PROPORTION = 0.05  # Reduced to only 5%

# -------------------------------------------------------------------------
# Optimization Model
# -------------------------------------------------------------------------
def optimize_rake_allocation(df):
    print("Building optimization model...")

    prob = LpProblem("Rake_Formation_Optimization", LpMinimize)

    y = {i: LpVariable(f"y_{i}", 0, 1, cat="Binary") for i in df.index}
    q = {i: LpVariable(f"q_{i}", 0, cat="Continuous") for i in df.index}

    # Debug: Print cost comparison for first few rows
    print("\n🔍 Cost Comparison (First 10 rows):")
    print("Index | Rail_Cost | Road_Cost | Rail_Advantage | On_Time_Prob | Feasible")
    for i, row in df.head(10).iterrows():
        planned_qty = float(row.get("planned_qty_t", 0) or 0)
        rail_cost_per_ton = float(row.get("pred_rail_cost_total", 0) or 0) / max(planned_qty, 1)
        road_cost_per_ton = float(row.get("pred_road_cost_total", 0) or 0) / max(planned_qty, 1)
        terminal_cost = float(row.get("terminal_cost", 0) or 0)
        demurrage = float(row.get("expected_demurrage", 0) or 0)
        on_time_prob = float(row.get("on_time_prob", 0) or 0)
        rake_avail = int(row.get("rake_available", 1))
        on_time_label = int(row.get("on_time_label", 1))
        min_rake_tonnage = float(row.get("min_rake_tonnage", 0) or 0)
        feasible = 1 if (rake_avail == 1 and on_time_label == 1 and planned_qty >= min_rake_tonnage) else 0
        
        rail_advantage = road_cost_per_ton - rail_cost_per_ton - (terminal_cost + demurrage)/max(planned_qty, 1)
        
        print(f"{i:5} | {rail_cost_per_ton:9.2f} | {road_cost_per_ton:9.2f} | {rail_advantage:13.2f} | {on_time_prob:11.2f} | {feasible:8}")

    objective = []
    for i, row in df.iterrows():
        planned_qty = float(row.get("planned_qty_t", 0) or 0)
        rail_cost = float(row.get("pred_rail_cost_total", 0) or 0)
        road_cost = float(row.get("pred_road_cost_total", 0) or 0)
        terminal_cost = float(row.get("terminal_cost", 0) or 0)
        demurrage = float(row.get("expected_demurrage", 0) or 0)
        prio = float(row.get("priority_score", 0) or 0)

        # Enhanced multi-objective cost with stronger rail incentives
        rail_cost_term = (rail_cost / max(planned_qty, 1)) * q[i]
        road_cost_term = (road_cost / max(planned_qty, 1)) * (planned_qty - q[i])
        fixed_costs = terminal_cost * y[i] + demurrage * y[i]

        # Much stronger rewards for reliability and priority
        on_time_prob = float(row.get("on_time_prob", 0))
        reliability_reward = RELIABILITY_WEIGHT * on_time_prob * planned_qty * y[i]  # Scaled by quantity
        priority_reward = PRIORITY_WEIGHT * prio * planned_qty * y[i]  # Scaled by quantity

        # Environmental/sustainability bonus for rail
        sustainability_bonus = 0.05 * planned_qty * y[i]  # Increased incentive for rail

        # Combine terms - making rail much more attractive
        objective.append(rail_cost_term + road_cost_term + fixed_costs - reliability_reward - priority_reward - sustainability_bonus)

    prob += lpSum(objective), "Total_Cost"

    # Count feasible rail options
    feasible_rail_indices = []
    for i, row in df.iterrows():
        planned_qty = float(row.get("planned_qty_t", 0) or 0)
        min_rake_tonnage = float(row.get("min_rake_tonnage", 0) or 0)
        rake_avail = int(row.get("rake_available", 1))
        on_time = int(row.get("on_time_label", 1))
        feasible = 1 if (rake_avail == 1 and on_time == 1 and planned_qty >= min_rake_tonnage) else 0

        if feasible == 0:
            prob += y[i] == 0, f"Feasible_{i}"
            prob += q[i] == 0, f"Qty_Zero_{i}"  # Also force quantity to zero
        else:
            feasible_rail_indices.append(i)

        prob += q[i] <= planned_qty * y[i], f"Qty_Upper_{i}"
        if min_rake_tonnage > 0 and feasible == 1:
            prob += q[i] >= min_rake_tonnage * y[i], f"Qty_Lower_{i}"

    # Add minimum rail utilization constraint (very soft version)
    if feasible_rail_indices:
        min_rail_orders = max(1, int(MIN_RAKE_PROPORTION * len(feasible_rail_indices)))
        print(f"🔧 Constraint: At least {min_rail_orders} out of {len(feasible_rail_indices)} feasible orders must use rail")
        
        # Create a two-stage approach: first try with constraint, then without if infeasible
        prob_temp = prob.copy()
        prob_temp += lpSum([y[i] for i in feasible_rail_indices]) >= min_rail_orders, "Min_Rail_Utilization"
        
        print("Solving with rail constraint...")
        prob_temp.solve(PULP_CBC_CMD(msg=False))
        
        if prob_temp.status == 1:  # Optimal
            print("✅ Solution found with rail constraint")
            prob = prob_temp
        else:
            print("⚠️  No solution with rail constraint, solving without...")
            # Solve without the rail constraint
            prob.solve(PULP_CBC_CMD(msg=False))

    # Yard capacity constraints (only if columns exist) - make them optional
    yard_constraints_added = 0
    if {"yard_id", "siding_slots"} <= set(df.columns):
        for yard, grp in df.groupby("yard_id"):
            slots = float(grp["siding_slots"].iloc[0])
            if slots > 0:
                # Make this a soft constraint by allowing some overflow
                slack = LpVariable(f"slack_siding_{yard}", 0, cat="Continuous")
                prob += lpSum([y[i] for i in grp.index]) <= slots + slack, f"SidingSlots_{yard}"
                # Add small penalty for using slack
                prob += slack * 1000  # Small penalty for exceeding slots
                yard_constraints_added += 1
        print(f"🔧 Added {yard_constraints_added} yard slot constraints (with slack)")

    inventory_constraints_added = 0
    if {"yard_id", "inventory_t", "production_forecast_t"} <= set(df.columns):
        for yard, grp in df.groupby("yard_id"):
            cap = float((grp["inventory_t"].fillna(0) + grp["production_forecast_t"].fillna(0)).iloc[0])
            if cap > 0:
                # Make this a soft constraint
                slack = LpVariable(f"slack_inv_{yard}", 0, cat="Continuous")
                prob += lpSum([q[i] for i in grp.index]) <= cap + slack, f"Inventory_{yard}"
                # Add small penalty for using slack
                prob += slack * 500  # Small penalty for exceeding inventory
                inventory_constraints_added += 1
        print(f"🔧 Added {inventory_constraints_added} inventory constraints (with slack)")

    print("Solving MILP...")
    prob.solve(PULP_CBC_CMD(msg=False))
    print(f"Solver Status: {LpStatus[prob.status]}")

    # Extract results
    if prob.status == 1:  # Optimal
        df["y_rail"] = [int(y[i].value()) if y[i].value() is not None else 0 for i in df.index]
        df["q_rail_tons"] = [float(q[i].value()) if q[i].value() is not None else 0.0 for i in df.index]
    else:
        print("❌ No optimal solution found. Using fallback...")
        # Fallback: assign rail to orders with positive rail advantage
        df["y_rail"] = 0
        df["q_rail_tons"] = 0.0
        
        for i, row in df.iterrows():
            planned_qty = float(row.get("planned_qty_t", 0) or 0)
            rail_cost_per_ton = float(row.get("pred_rail_cost_total", 0) or 0) / max(planned_qty, 1)
            road_cost_per_ton = float(row.get("pred_road_cost_total", 0) or 0) / max(planned_qty, 1)
            terminal_cost = float(row.get("terminal_cost", 0) or 0)
            demurrage = float(row.get("expected_demurrage", 0) or 0)
            rake_avail = int(row.get("rake_available", 1))
            on_time_label = int(row.get("on_time_label", 1))
            min_rake_tonnage = float(row.get("min_rake_tonnage", 0) or 0)
            
            rail_advantage = road_cost_per_ton - rail_cost_per_ton - (terminal_cost + demurrage)/max(planned_qty, 1)
            feasible = 1 if (rake_avail == 1 and on_time_label == 1 and planned_qty >= min_rake_tonnage) else 0
            
            if feasible == 1 and rail_advantage > 0:
                df.loc[i, "y_rail"] = 1
                df.loc[i, "q_rail_tons"] = planned_qty

    df["optimized_total_cost"] = [
        (row.get("pred_rail_cost_total", 0) * df.loc[i, "q_rail_tons"] / max(row.get("planned_qty_t", 1), 1))
        + row.get("terminal_cost", 0) * df.loc[i, "y_rail"]
        + row.get("expected_demurrage", 0) * df.loc[i, "y_rail"]
        + row.get("pred_road_cost_total", 0) * (row.get("planned_qty_t", 1) - df.loc[i, "q_rail_tons"]) / max(row.get("planned_qty_t", 1), 1)
        for i, row in df.iterrows()
    ]

    df["optimized_mode"] = np.where(df["y_rail"] == 1, "Rail", "Road")

    # Print optimization summary
    total_orders = len(df)
    rail_orders = df["y_rail"].sum()
    road_orders = total_orders - rail_orders
    rail_tonnage = df["q_rail_tons"].sum()
    total_tonnage = df["planned_qty_t"].sum()
    
    print(f"\n📊 Optimization Summary:")
    print(f"   Total orders: {total_orders}")
    print(f"   Rail orders: {rail_orders} ({rail_orders/total_orders*100:.1f}%)")
    print(f"   Road orders: {road_orders} ({road_orders/total_orders*100:.1f}%)")
    if total_tonnage > 0:
        print(f"   Rail tonnage: {rail_tonnage:.0f} tons ({rail_tonnage/total_tonnage*100:.1f}% of total)")
    print(f"   Total cost: {df['optimized_total_cost'].sum():,.0f}")

    # Enhanced file saving with error handling
    current_output_path = OUTPUT_PATH
    try:
        df.to_csv(current_output_path, index=False)
        print(f"✅ Optimization completed and saved to {current_output_path}")
    except PermissionError:
        # If file is locked, try with a timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        current_output_path = f"C:\\Users\\INDIA\\Desktop\\sihPS2\\optimized_rake_allocations_{timestamp}.csv"
        df.to_csv(current_output_path, index=False)
        print(f"⚠️ Original file was locked. Saved to: {current_output_path}")
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        print("Displaying results instead...")
    
    return df


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    print("🔹 Loading pipeline and dataset...")
    
    # Check if input file exists
    if not os.path.exists(DATA_PATH):
        print(f"❌ Input file not found: {DATA_PATH}")
        return
    
    # Check if model directory exists
    if not os.path.exists(MODEL_DIR):
        print(f"❌ Model directory not found: {MODEL_DIR}")
        print("Please run training first using multi_model.py")
        return
    
    pipeline = MultiModelPipeline(DATA_PATH, MODEL_DIR)
    df = pd.read_csv(DATA_PATH)
    
    # Recreate engineered date features (must match training logic)
    if "plan_date" in df.columns:
        try:
            df["plan_date"] = pd.to_datetime(df["plan_date"], dayfirst=True)
            df["plan_dayofweek"] = df["plan_date"].dt.dayofweek
            df["plan_month"] = df["plan_date"].dt.month
            df["plan_day"] = df["plan_date"].dt.day
        except Exception as e:
            print(f"⚠️ Date feature creation failed: {e}")

    # Load saved feature list
    feature_path = os.path.join(MODEL_DIR, "feature_cols.pkl")
    if os.path.exists(feature_path):
        pipeline.feature_cols = joblib.load(feature_path)
        print(f"Loaded {len(pipeline.feature_cols)} feature columns for prediction.")
    else:
        raise ValueError("Missing feature_cols.pkl — run training first using multi_model.py")

    # Run predictions
    preds = pipeline.predict_all(df)

    df_pred = df.copy()
    for k, v in preds.items():
        df_pred[k] = v

    result_df = optimize_rake_allocation(df_pred)

    print("\n✅ Optimization completed.")
    print("Sample results:")
    sample_results = result_df[["plan_id", "optimized_mode", "q_rail_tons", "optimized_total_cost"]].head(10)
    print(sample_results)
    
    # Show some rail assignments
    rail_assignments = result_df[result_df["optimized_mode"] == "Rail"]
    if len(rail_assignments) > 0:
        print(f"\n🚆 Rail Assignments ({len(rail_assignments)} total):")
        print(rail_assignments[["plan_id", "optimized_mode", "q_rail_tons", "optimized_total_cost"]].head(10))
    else:
        print("\n❌ No rail assignments found!")


if __name__ == "__main__":
    main()