import re
import random
import ast


def extract_solution(solution_str):
    """Extract the solution grid from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def parse_grid(grid_str):
    """Parse a 9x9 grid string into a 2D list of integers.
    
    Args:
        grid_str: A string representation of a 9x9 grid in the format
                 '[[n1,n2,...,n9], [n1,n2,...,n9], ...]'
                 where n1-n9 are integers or '*' for empty cells
    
    Returns:
        A 2D list of integers where '*' is converted to 0, or None if parsing fails
    """
    try:
        # Remove whitespace and split by commas
        grid_str = grid_str.replace(' ', '')
        rows = grid_str.strip('[]').split('],[')
        
        grid = []
        for row in rows:
            # Remove brackets and split by comma
            row = row.strip('[]')
            numbers = row.split(',')
            # Convert to integers, handling '*' as 0
            row_numbers = [0 if n == '*' else int(n) for n in numbers]
            if len(row_numbers) != 9:
                return None
            grid.append(row_numbers)
            
        if len(grid) != 9:
            return None
            
        return grid
    except (ValueError, IndexError):
        return None


def validate_solution(solution_grid, puzzle_grid):
    """Validate that the solution is valid and matches the puzzle constraints.
    
    Args:
        solution_grid: A 9x9 grid of integers representing the solution
        puzzle_grid: A 9x9 grid of integers representing the original puzzle
        
    Returns:
        list[bool, bool]: [is_valid, is_correct]
    """
    if solution_grid is None or puzzle_grid is None:
        return [False, False]
        
    # Check dimensions
    if len(solution_grid) != 9 or len(solution_grid[0]) != 9:
        return [False, False]
        
    # Check that solution matches puzzle constraints
    for i in range(9):
        for j in range(9):
            if puzzle_grid[i][j] != 0 and solution_grid[i][j] != puzzle_grid[i][j]:
                return [False, False]
                
    # Check rows
    for row in solution_grid:
        if sorted(row) != [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            return [True, False]
            
    # Check columns
    for j in range(9):
        col = [solution_grid[i][j] for i in range(9)]
        if sorted(col) != [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            return [True, False]
            
    # Check 3x3 boxes
    for box_i in range(0, 9, 3):
        for box_j in range(0, 9, 3):
            box = []
            for i in range(3):
                for j in range(3):
                    box.append(solution_grid[box_i + i][box_j + j])
            if sorted(box) != [1, 2, 3, 4, 5, 6, 7, 8, 9]:
                return [True, False]
                
    return [True, True]


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for sudoku task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing puzzle and solution grids
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    puzzle = ground_truth['puzzle']
    solution = ground_truth['solution']

    puzzle_grid = []
    for i in range(9):
        puzzle_grid.append(list([int(n) for n in puzzle[i*9:(i+1)*9]]))
    
    response_grid_str = extract_solution(solution_str=solution_str)
    response_grid = parse_grid(response_grid_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Puzzle: {puzzle_grid}")
        print(f"Response: {response_grid}")

    if response_grid_str is None or response_grid is None:
        if do_print:
            print(f"No solution found")
        return 0
    
    # Validate solution
    valid, correct = validate_solution(response_grid, puzzle_grid)
    if valid and correct:
        return score
    elif valid:
        return format_score
    else:
        return 0