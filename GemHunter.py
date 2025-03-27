import time
import os
from pysat.solvers import Solver
import itertools
import copy

def readMap(filename):
    map = []
    with open(filename) as file:
        lines = file.readlines()
        for line in lines:
            row = line.replace("\n", "").replace(" ", "").split(",")
            rowMap = []
            for element in row:
                rowMap.append(element)  # Keep as string to distinguish between numbers and empty cells
            map.append(rowMap)
    return map

def getAdjacentCells(x, y, rows, cols):
    """Get all adjacent cells (including diagonals) for a given position."""
    adjacent = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue  # Skip the cell itself
            nx, ny = x + dx, y + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                adjacent.append((ny, nx))  # Return in (row, col) format
    return adjacent

def get_at_least_n_constraint(variables, n):
    """Get clauses to ensure at least n variables are true."""
    clauses = []
    if n <= 0:
        return clauses  # Always satisfied
    
    # Handle case where n is greater than len(variables)
    if n > len(variables):
        clauses.append([])  # Impossible constraint - add empty clause (contradiction)
        return clauses
    
    # Generate all combinations of (len(variables) - n + 1) negative literals
    for neg_lits in itertools.combinations(variables, len(variables) - n + 1):
        clause = [var for var in neg_lits]  # We want at least one to be true, so we add positives
        clauses.append(clause)
    
    return clauses

def get_at_most_n_constraint(variables, n):
    """Get clauses to ensure at most n variables are true."""
    clauses = []
    if n >= len(variables):
        return clauses  # Always satisfied
    
    # Generate all combinations of (n + 1) positive literals
    for pos_lits in itertools.combinations(variables, n + 1):
        clause = [-var for var in pos_lits]  # We want at least one to be false, so we add negatives
        clauses.append(clause)
    
    return clauses

def get_exactly_n_constraint(variables, n):
    """Get clauses to ensure exactly n variables are true."""
    clauses = []
    
    # Handle edge cases
    if n < 0:
        clauses.append([])  # Impossible constraint - add empty clause
        return clauses
    
    if n == 0:
        # All variables must be false
        for var in variables:
            clauses.append([-var])
        return clauses
    
    if n > len(variables):
        clauses.append([])  # Impossible constraint - add empty clause
        return clauses
    
    # At least n variables are true
    at_least_n_clauses = get_at_least_n_constraint(variables, n)
    clauses.extend(at_least_n_clauses)
    
    # At most n variables are true
    at_most_n_clauses = get_at_most_n_constraint(variables, n)
    clauses.extend(at_most_n_clauses)
    
    return clauses

def generateCNFs(map, var_map):
    """Create CNF constraints based on the map."""
    clauses = []
    rows = len(map)
    cols = len(map[0])
    
    # First, create variable mapping for each cell
    var_idx = 1  # Variables in PySAT are 1-indexed
    for i in range(rows):
        for j in range(cols):
            if map[i][j] == "_":  # Empty cell (unknown trap/gem)
                var_map[(i, j)] = var_idx
                var_idx += 1
    
    # Generate constraints for numbered cells
    for i in range(rows):
        for j in range(cols):
            cell_value = map[i][j]
            if cell_value != "_" and cell_value.isdigit():  # Cell contains a number
                number = int(cell_value)
                adjacent_cells = getAdjacentCells(j, i, rows, cols)
                
                # Get variable numbers for adjacent cells that are unknown (0)
                vars_surrounding = []
                for r, c in adjacent_cells:
                    if map[r][c] == "_":  # Only unknown cells
                        vars_surrounding.append(var_map[(r, c)])
                
                # Add constraints for exactly N traps in surrounding cells
                exactly_n_clauses = get_exactly_n_constraint(vars_surrounding, number)
                clauses.extend(exactly_n_clauses)
    
    # Remove duplicate clauses
    unique_clauses = [list(x) for x in set(tuple(sorted(x)) for x in clauses)]
    return unique_clauses

def solve_with_sat(map):
    """Solve the trap/gem puzzle using PySAT."""
    var_map = {}  # Maps (row, col) to variable number
    
    # Generate CNF clauses
    cnf_clauses = generateCNFs(map, var_map)
    
    # Create a reverse mapping from variable to position
    pos_map = {var: pos for pos, var in var_map.items()}
    
    # Solve using PySAT
    solver = Solver(name='g4')
    for clause in cnf_clauses:
        solver.add_clause(clause)
    
    # Check if satisfiable
    if solver.solve():
        model = solver.get_model()
        result_map = [row[:] for row in map]  # Copy the original map
        
        # Interpret the solution
        for var, val in enumerate(model, 1):
            if val > 0 and val in pos_map:  # Positive literal (true) = trap
                r, c = pos_map[val]
                result_map[r][c] = 'T'  # Trap
            elif val < 0 and -val in pos_map:  # Negative literal (false) = gem
                r, c = pos_map[-val]
                result_map[r][c] = 'G'  # Gem
        
        return result_map
    else:
        return None  # No solution exists

def pl_resolve(ci, cj):
    """Resolve two clauses and return the result."""
    resolvents = []
    
    # Look for complementary literals
    for lit_i in ci:
        if -lit_i in cj:  # Found complementary literals
            # Create a new clause with all literals except the complementary ones
            new_clause = [l for l in ci if l != lit_i] + [l for l in cj if l != -lit_i]
            # Remove duplicates
            new_clause = list(set(new_clause))
            resolvents.append(new_clause)
    
    return resolvents

def pl_resolution_algo(clauses):
    """PL-RESOLUTION algorithm."""
    start_time = time.time()
    timeout_seconds = 120  # 2 minutes timeout
    all_clauses = copy.deepcopy(clauses)
    
    while True:
        new_clauses = []
        
        # Try all pairs of clauses
        for i in range(len(all_clauses)):
            for j in range(i + 1, len(all_clauses)):
                # Kiểm tra timeout
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                if elapsed_time > timeout_seconds:
                    print(f" - PL-Resolution timed out after {elapsed_time:.2f} seconds")
                    return False 
                
                resolvents = pl_resolve(all_clauses[i], all_clauses[j])
                
                # Check for empty clause (contradiction)
                if [] in resolvents:
                    return False  # UNSAT
                
                # Add new resolvents
                for resolvent in resolvents:
                    if resolvent not in all_clauses and resolvent not in new_clauses:
                        new_clauses.append(resolvent)
        
        # If no new clauses were added, we're done
        if all(clause in all_clauses for clause in new_clauses):
            return True  # SAT
        
        # Add new clauses to the set
        all_clauses.extend(new_clauses)

def estimate_bruteforce_time(total_cases):
    """Estimate bruteforce solving time based on observed performance"""

    # For 5x5 map with 11 empty cells: 2048 cases in 6.834 milliseconds
    base_cases = 2048
    base_time = 6.834  # milliseconds
    observed_time_per_cases = base_time / base_cases
    
    # Estimate total time
    total_time_ms = float(total_cases) * observed_time_per_cases
    
    return total_time_ms

def pl_resolution_solve(map):
    """
    Solve the trap/gem puzzle using PL-RESOLUTION algorithm.
    
    Args:
    - map: The input map to solve
    - timeout_minutes: Maximum time to search before stopping (default 5 minutes)
    
    Returns:
    - result_map: Solved map or None if no solution found
    - elapsed_time: Time taken to search
    - timed_out: Boolean indicating if search was stopped due to timeout
    """
    start_time = time.time()
    var_map = {}  # Maps (row, col) to variable number
    
    # Generate CNF clauses
    clauses = generateCNFs(map, var_map)
    
    # Create a reverse mapping from variable to position
    pos_map = {var: pos for pos, var in var_map.items()}

    # Get empty cells (unknown cells)
    empty_cells = list(var_map.keys())
    length = len(empty_cells)
    
    end = 1 << length
    print(f" - Bruteforcing {end} cases for {length} empty cells...")

    # Use PL-RESOLUTION to check satisfiability
    is_sat = pl_resolution_algo(clauses)
    
    # If satisfiable, find a model using brute force
    if is_sat:
        # Get all variables
        variables = list(var_map.values())
        total_variables = len(variables)
        
        # Try all possible truth assignments
        for i in range(2 ** total_variables):
            
            # Create assignment
            assignment = {}
            for j, var in enumerate(variables):
                # Use bits of i to determine variable's truth value
                assignment[var] = (i & (1 << j)) > 0
            
            # Check if this assignment satisfies all clauses
            satisfied = True
            for clause in clauses:
                clause_satisfied = False
                for lit in clause:
                    var = abs(lit)
                    is_negated = lit < 0
                    truth_value = assignment[var]
                    if is_negated:
                        truth_value = not truth_value
                    if truth_value:
                        clause_satisfied = True
                        break
                if not clause_satisfied:
                    satisfied = False
                    break
            
            if satisfied:
                # Create solution map
                result_map = [row[:] for row in map]
                for var, is_true in assignment.items():
                    if var in pos_map:
                        r, c = pos_map[var]
                        result_map[r][c] = 'T' if is_true else 'G'
                
                elapsed_time = time.time() - start_time
                return result_map, elapsed_time, False  # Solution found, not timed out
    else:
        expected_total_time = estimate_bruteforce_time(end)
        return None, expected_total_time, True
    
    elapsed_time = time.time() - start_time
    return None, elapsed_time, False  # No solution, not timed out

def dpll_solve(map):
    """Solve the trap/gem puzzle using DPLL algorithm."""
    start_time = time.time()
    var_map = {}  # Maps (row, col) to variable number
    
    # Generate CNF clauses
    clauses = generateCNFs(map, var_map)
    
    # Create a reverse mapping from variable to position
    pos_map = {var: pos for pos, var in var_map.items()}
    
    # Convert clauses to a list of lists for the new implementation
    KB = [list(clause) for clause in clauses]
    
    def has_unit_clause(KB):
        return any(len(clause) == 1 for clause in KB)
    
    def choose_unit_clause(KB):
        for clause in KB:
            if len(clause) == 1:
                return clause[0]
        return None
    
    def propagate_unit(KB, unit):
        KB_copy = [row.copy() for row in KB]
        KB_copy = [clause for clause in KB_copy if unit not in clause]
        
        for clause in KB_copy:
            if -unit in clause:
                clause.remove(-unit)
        
        return KB_copy
    
    def choose_literal(KB):
        return KB[0][0]
    
    def dpll(KB, model):
        # If KB is empty, return model
        if len(KB) == 0:
            return model
        
        # If any clause is empty, return None
        if any(len(clause) == 0 for clause in KB):
            return None
        
        # Unit propagation
        while has_unit_clause(KB):
            unit = choose_unit_clause(KB)
            KB = propagate_unit(KB, unit)
            model.append(unit)
            
            # If KB is empty, return model
            if len(KB) == 0:
                return model
            
            # If any clause is empty, return None
            if any(len(clause) == 0 for clause in KB):
                return None
        
        # Choose a literal
        literal = choose_literal(KB)
        
        # Try True assignment
        KB_copy1 = propagate_unit(KB.copy(), literal)
        result = dpll(KB_copy1, model + [literal])
        if result is not None:
            return result
        
        # Try False assignment
        KB_copy2 = propagate_unit(KB.copy(), -literal)
        result = dpll(KB_copy2, model + [-literal])
        if result is not None:
            return result
        
        # No solution found
        return None
    
    # Call DPLL
    model = dpll(KB, [])
    
    # If a solution is found, create the result map
    if model:
        result_map = [row[:] for row in map]
        for var in sorted(var_map.values()):
            if var in pos_map:
                r, c = pos_map[var]
                # Determine trap/gem based on the model
                result_map[r][c] = 'T' if var in model else 'G'
        
        elapsed_time = time.time() - start_time
        return result_map, elapsed_time
    
    elapsed_time = time.time() - start_time
    return None, elapsed_time


def display_map(map):
    """Display the map in a readable format."""
    for row in map:
        print(", ".join(str(cell) for cell in row))

def write_solution_to_file(filename, solution):
    """Write the solution to a file."""
    with open(filename, 'w') as f:
        for row in solution:
            f.write(', '.join(row) + '\n')

def main():
    # Locate test cases in the testcases directory
    testcases_dir = 'testcases'
    input_files = [f for f in os.listdir(testcases_dir) if f.startswith('input_') and f.endswith('.txt')]
    
    # Process each input file
    for input_file in sorted(input_files):
        file_num = input_file.split('_')[1].split('.')[0]
        output_file = f'output_{file_num}.txt'
        
        input_path = os.path.join(testcases_dir, input_file)
        output_path = os.path.join(testcases_dir, output_file)
        
        print(f"\nSolving test case {file_num} from {input_path}...")
        map = readMap(input_path)
        
        print("Original Map:")
        display_map(map)
        print("\n")
        
        # Solve using PySAT (reference solution)
        print("Solving using PySAT...")
        start_time = time.time()
        sat_solution = solve_with_sat(map)
        sat_time = time.time() - start_time
        
        if sat_solution:
            print("Solution found with PySAT in {:.6f} seconds:".format(sat_time))
            display_map(sat_solution)
        else:
            print("No solution found with PySAT.")
        
        print("\n")
        
        # Solve using DPLL (backtracking)
        print("Solving using DPLL (Backtracking)...")
        dpll_solution, dpll_time = dpll_solve(map)
        
        if dpll_solution:
            print("Solution found with DPLL in {:.6f} seconds:".format(dpll_time))
            display_map(dpll_solution)
        else:
            print("No solution found with DPLL.")
        
        print("\n")
        
        # Solve using PL-RESOLUTION (brute force)
        print("Solving using PL-RESOLUTION (Brute Force)...")
        pl_solution, pl_time, pl_time_out = pl_resolution_solve(map)
        
        if pl_time_out:
            print("PL-RESOLUTION search timed out.")
        else:
            if pl_solution:
                print("Solution found with PL-RESOLUTION in {:.6f} seconds:".format(pl_time))
                display_map(pl_solution)
            else:
                print("No solution found with PL-RESOLUTION.")
        
        # Compare times
        print("\nPerformance Comparison:")
        print("PySAT: {:.6f} miliseconds".format(sat_time * 1000))
        print("DPLL (Backtracking): {:.6f} miliseconds".format(dpll_time * 1000))
        if pl_time_out:
            print("PL-RESOLUTION (Brute Force) [Estimated Time]: {:.6f} miliseconds".format(pl_time * 1000))
        else:    
            print("PL-RESOLUTION (Brute Force): {:.6f} miliseconds".format(pl_time * 1000))

        # Write solution to output file
        if sat_solution:
            write_solution_to_file(output_path, sat_solution)
            print(f"Solution written to {output_path}")
        else:
            print(f"No solution found for {input_file}")

if __name__ == "__main__":
    main()