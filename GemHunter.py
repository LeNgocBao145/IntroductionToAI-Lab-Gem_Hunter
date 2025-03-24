import time
from pysat.solvers import Solver
import itertools

# 1. Assign a logical variable for each cell in the map: True (T): The cell contains a trap. False (G): The cell contains
# a gem.
    # Each tile with a number represents the number of traps surrounding it. (Number from 1 - 8)
def readMap(filename):
    map = []
    with open(filename) as file:
        lines = file.readlines()
        for line in lines:
            row = line.replace("\n", "").replace(" ", "").replace("_", "0").split(",")
            rowMap = []
            for element in row:
                rowMap.append(int(element))
            map.append(rowMap)
    return map
print(readMap("testcases/input_1.txt"))
# 2. (Report) Write constraints for cells containing numbers to obtain a set of constraint clauses in CNF (note that
# you need to remove duplicate clauses)
def getUnkTile(x, y, map):
    row = len(map)
    col = len(map[0])
    unkTile = []

    if x > 0:
        if map[y][x - 1] == 0:
            unkTile.append(map[y][x - 1])
    if x < col - 1:
        if map[y][x + 1] == 0:
            unkTile.append(map[y][x + 1])
    if y > 0:
        if map[y - 1][x] == 0:
            unkTile.append(map[y - 1][x])
        if x < col - 1:
            if map[y - 1][x + 1] == 0:
                unkTile.append(map[y - 1][x + 1])
        if x > 0:
            if map[y - 1][x - 1] == 0:
                unkTile.append(map[y - 1][x - 1])
    if y < row - 1:
        if map[y + 1][x] == 0:
            unkTile.append(map[y + 1][x])
        if x < col - 1:
            if map[y + 1][x + 1] == 0:
                unkTile.append(map[y + 1][x + 1])
        if x > 0:
            if map[y + 1][x - 1] == 0:
                unkTile.append(map[y + 1][x - 1])

    return unkTile
        
                
var_map = {}
# 3. (Implement) Generate CNFs automatically.
def generateCNFs(map):
    """Create CNF constraints based on the map."""
    clauses = []
    rows = len(map)
    cols = len(map[0])
    for i in range(rows):
        for j in range(cols):
            cell_value = map[i][j]
            if cell_value != 0 and cell_value.isdigit():
                # Cell contains a number
                number = int(cell_value)
                surrounding_cells = getUnkTile(i, j)
                
                # Get variable numbers for surrounding cells
                vars_surrounding = [var_map[(r, c)] for r, c in surrounding_cells]
                
                # Add constraints for exactly N traps in surrounding cells
                exactly_n_clauses = get_exactly_n_constraint(vars_surrounding, number)
                clauses.extend(exactly_n_clauses)
    
    unique_clauses = [list(x) for x in set(tuple(x) for x in clauses)]
    return unique_clauses
    # return clauses
    
def get_exactly_n_constraint( variables, n):
    """Get clauses to ensure exactly n variables are true."""
    clauses = []
    # At least n variables are true
    at_least_n_clauses = get_at_least_n_constraint(variables, n)
    clauses.extend(at_least_n_clauses)
    
    # At most n variables are true
    at_most_n_clauses = get_at_most_n_constraint(variables, n)
    clauses.extend(at_most_n_clauses)
    
    return clauses

def get_at_least_n_constraint( variables, n):
    """Get clauses to ensure at least n variables are true."""
    clauses = []
    if n <= 0:
        return clauses  # Always satisfied
    
    # Generate all combinations of (len(variables) - n + 1) negative literals
    for neg_lits in itertools.combinations(variables, len(variables) - n + 1):
        clause = [var for var in neg_lits]
        clauses.append(clause)
    
    return clauses

def get_at_most_n_constraint( variables, n):
    """Get clauses to ensure at most n variables are true."""
    clauses = []
    if n >= len(variables):
        return clauses  # Always satisfied
    
    # Generate all combinations of (n + 1) positive literals
    for pos_lits in itertools.combinations(variables, n + 1):
        clause = [-var for var in pos_lits]
        clauses.append(clause)
    
    return clauses

# 4. (Implement) Using the pysat library to find the value for each variable and infer the result.


# 5. (Implement) Program brute-force and backtracking algorithm to compare their speed (by measuring running
# time, which is how long it takes for a computer to perform a specific task) and their performance with using the
# library

# function PL-RESOLUTION(KB,α) returns true or false
# Brute-force algorithm
# def PL_RESOLUTION():
#  inputs: KB, the knowledge base, a sentence in propositional logic
#  α, the query, a sentence in propositional logic
#  clauses ← the set of clauses in the CNF representation of KB ∧ ¬α
#  new ← { }
#  loop do
#  for each pair of clauses Ci , Cj in clauses do
#  resolvents ← PL-RESOLVE(Ci , Cj)
#  if resolvents contains the empty clause then return true
#  new ← new ∪ resolvents
#  if new ⊆ clauses then return false
#  clauses ← clauses ∪ new


# function DPLL-SATISFIABLE?(s) returns true or false
#  inputs: s, a sentence in propositional logic
#  clauses ← the set of clauses in the CNF representation of s
#  symbols ← a list of the proposition symbols in s
#  return DPLL(clauses, symbols,{ })
 
#  function DPLL(clauses, symbols, model) returns true or false
#  if every clause in clauses is true in model then return true
#  if some clause in clauses is false in model then return false
#  P, value ← FIND-PURE-SYMBOL(symbols, clauses, model)
#  if P is non-null then return DPLL(clauses, symbols – P, model ∪ {P=value})
#  P, value ← FIND-UNIT-CLAUSE(clauses, model)
#  if P is non-null then return DPLL(clauses, symbols – P, model ∪ {P=value})
#  P ← FIRST(symbols); rest ← REST(symbols)
#  return DPLL(clauses, rest, model ∪ {P=true}) or  DPLL(clauses, rest, model ∪ {P=false}))