from pulp import *
import numpy as np
import json

def solve(data):

    TotalAircraft = data["TotalAircraft"]
    TotalRoutes = data["TotalRoutes"]
    Availability = np.array(data["Availability"])
    Demand = np.array(data["Demand"])
    Capacity = np.array(data["Capacity"])
    Costs = np.array(data["Costs"])

    prob = LpProblem("Aircraft Assignment", LpMinimize)
    # 定义决策变量
    assignment_vars = LpVariable.dicts("Assignment", (range(TotalAircraft), range(TotalRoutes)), cat='Binary')

    # 定义目标函数
    prob += lpSum([Costs[i][j] * assignment_vars[i][j] for i in range(TotalAircraft) for j in range(TotalRoutes)])

    # 添加约束条件
    for i in range(TotalAircraft):
        prob += lpSum([assignment_vars[i][j] for j in range(TotalRoutes)]) <= Availability[i]

    for j in range(TotalRoutes):
        prob += lpSum([assignment_vars[i][j] for i in range(TotalAircraft)]) == Demand[j]

    for i in range(TotalAircraft):
        for j in range(TotalRoutes):
            prob += assignment_vars[i][j] <= Capacity[i][j]

    prob.solve()

    if prob.status == LpStatusOptimal:
        return value(prob.objective)
    else:
        return LpStatus[prob.status]


with open("data.json", 'r') as f:
    data = json.load(f)
answer={}
answer["obj"]=solve(data)
with open("../answer.json", 'w') as f:
    json.dump(answer, f, indent=4)