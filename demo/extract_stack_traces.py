"""Extract all stack traces and solutions from developer log files and save to CSV."""

import re
import sys
from pathlib import Path
import pandas as pd


def extract_stack_traces_and_solutions(log_file_path: str) -> list[dict]:
    """Extract all stack traces and their solutions from a developer log file.

    Args:
        log_file_path: Path to the developer log file

    Returns:
        List of dicts with keys: 'stack_trace', 'solution'
    """
    results = []
    current_trace = []
    current_solution = []
    in_trace = False
    in_solution = False
    solution_format = None  # 'json' or 'text'

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Check if we're starting a new stack trace
            if "Traceback (most recent call last):" in line:
                # If already in a trace, this is a chained exception - continue collecting
                if in_trace and len(current_trace) > 0:
                    trace_start = line.find("Traceback (most recent call last):")
                    current_trace.append(line[trace_start:] if trace_start != -1 else line)
                    continue

                # Save previous trace + solution if any (only if not currently in a trace)
                if current_trace and not in_trace:
                    results.append({
                        'stack_trace': ''.join(current_trace),
                        'solution': ''.join(current_solution).strip() if current_solution else ''
                    })

                # Start new trace
                current_trace = []
                current_solution = []
                in_trace = True
                in_solution = False
                solution_format = None

                # Extract the trace portion after the log prefix
                trace_start = line.find("Traceback (most recent call last):")
                current_trace.append(line[trace_start:])
                continue

            # Check if we're starting a solution
            if "This is how you can fix the error:" in line:
                in_trace = False
                in_solution = True
                # Don't include this line in the trace
                continue

            # If we're in a solution, detect format and collect lines
            if in_solution:
                stripped_line = line.strip()

                # Detect format on first line after "This is how you can fix the error:"
                if solution_format is None:
                    if stripped_line.startswith("```json"):
                        solution_format = 'json'
                        continue  # Skip the ```json marker
                    else:
                        solution_format = 'text'

                # Handle JSON format
                if solution_format == 'json':
                    if stripped_line.startswith("```") and len(current_solution) > 0:
                        # End of JSON solution
                        in_solution = False
                        continue
                    else:
                        current_solution.append(line)

                # Handle plain text format
                elif solution_format == 'text':
                    # End conditions for plain text solutions:
                    # 1. Another stack trace starts
                    # 2. Log line that's not part of solution (e.g., DEBUG, INFO)
                    # 3. Empty line followed by non-solution content
                    if "Traceback (most recent call last):" in line:
                        # New trace starting, save current and reset
                        results.append({
                            'stack_trace': ''.join(current_trace),
                            'solution': ''.join(current_solution).strip()
                        })
                        current_trace = []
                        current_solution = []
                        in_solution = False
                        in_trace = True
                        solution_format = None
                        trace_start = line.find("Traceback (most recent call last):")
                        current_trace.append(line[trace_start:])
                        continue
                    elif re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} (DEBUG|INFO)', line):
                        # Log line starting, solution ended
                        in_solution = False
                        continue
                    else:
                        # Part of solution
                        current_solution.append(line)
                continue

            # If we're in a trace, collect lines
            if in_trace:
                stripped_line = line.strip()

                # The trace ends ONLY at ERROR conda.cli.main_run line
                if stripped_line.startswith("ERROR conda.cli.main_run"):
                    # This is the final line of the trace
                    current_trace.append(line)
                    in_trace = False
                else:
                    # Everything else is part of the trace (including error lines, chaining messages, etc.)
                    if line.strip():  # Non-empty line
                        current_trace.append(line)
                    elif current_trace:  # Empty line after we've started collecting
                        current_trace.append(line)

    # Handle last trace + solution if file ends
    if current_trace:
        results.append({
            'stack_trace': ''.join(current_trace),
            'solution': ''.join(current_solution).strip() if current_solution else ''
        })

    return results


def clean_stack_trace(trace: str) -> str:
    """Clean a stack trace by removing log prefixes and keeping only the trace.

    Args:
        trace: Raw stack trace string with log prefixes

    Returns:
        Cleaned stack trace string
    """
    lines = trace.split('\n')
    cleaned_lines = []

    for line in lines:
        # Find where the actual trace content starts (after log timestamp/prefix)
        # Pattern: "2025-11-03 17:31:33,440 DEBUG [agents.developer...] "
        if "Traceback (most recent call last):" in line:
            start = line.find("Traceback (most recent call last):")
            cleaned_lines.append(line[start:])
        elif "File \"" in line or line.strip().startswith("^"):
            # Stack frame line or pointer line
            cleaned_lines.append(line)
        elif re.match(r'^[A-Z]\w*Error:', line.strip()):
            # Error line
            cleaned_lines.append(line)
        elif "ERROR conda.cli" in line:
            cleaned_lines.append(line)
        elif line.strip():
            # Other content line (might be error message continuation)
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def clean_solution(solution: str) -> str:
    """Clean a solution by removing extra whitespace and normalizing format.

    Args:
        solution: Raw solution string

    Returns:
        Cleaned solution string
    """
    # If it's JSON, try to parse and re-serialize for clean formatting
    if solution.strip().startswith('{'):
        try:
            import json
            parsed = json.loads(solution)
            # Extract just the reasoning_and_solution field if it exists
            if 'reasoning_and_solution' in parsed:
                return parsed['reasoning_and_solution']
            # Otherwise return the whole JSON as formatted string
            return json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            pass

    # For plain text, just clean up whitespace
    return solution.strip()


def find_developer_logs(task_dir: str) -> list[Path]:
    """Find all developer log files recursively in task directory.

    Args:
        task_dir: Path to task directory

    Returns:
        List of Path objects for developer log files
    """
    task_path = Path(task_dir)

    # Pattern: developer_<number>_<number>.txt or developer_<number>_<number>_ens.txt
    pattern1 = "developer_*_*.txt"
    pattern2 = "developer_*_*_ens.txt"

    # Find all matching files recursively
    log_files = []
    log_files.extend(task_path.rglob(pattern1))
    log_files.extend(task_path.rglob(pattern2))

    # Remove duplicates and sort
    log_files = sorted(set(log_files))

    return log_files


def extract_all_stack_traces(task_dir: str, output_csv: str, only_with_solutions: bool = False):
    """Extract stack traces and solutions from all developer logs and save to CSV.

    Args:
        task_dir: Path to task directory
        output_csv: Path to output CSV file
        only_with_solutions: If True, only save traces that have solutions
    """
    print(f"Searching for developer logs in: {task_dir}")
    log_files = find_developer_logs(task_dir)

    print(f"Found {len(log_files)} developer log file(s)")

    all_data = []
    trace_id = 0
    skipped_count = 0

    for log_file in log_files:
        print(f"Processing: {log_file}")
        results = extract_stack_traces_and_solutions(str(log_file))

        for result in results:
            cleaned_trace = clean_stack_trace(result['stack_trace'])
            cleaned_solution = clean_solution(result['solution']) if result['solution'] else ''

            # Skip traces without solutions if filtering is enabled
            if only_with_solutions and (not cleaned_solution or len(cleaned_solution) < 50):
                skipped_count += 1
                continue

            all_data.append({
                'id': trace_id,
                'stack_trace': cleaned_trace.strip(),
                'solution': cleaned_solution
            })
            trace_id += 1

        print(f"  Extracted {len(results)} stack trace(s) with solution(s)")

    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)

    print(f"\nTotal stack traces extracted: {len(all_data)}")
    print(f"Solutions found: {sum(1 for d in all_data if d['solution'])}")
    if only_with_solutions:
        print(f"Skipped (no solution): {skipped_count}")
    print(f"Saved to: {output_csv}")


def main():
    """Main function to extract and save stack traces with solutions."""
    if len(sys.argv) < 2:
        print("Usage: python extract_stack_traces.py <task_dir> [output_csv] [--only-with-solutions]")
        print("Example: python extract_stack_traces.py task/ stack_traces_with_solutions.csv")
        print("         python extract_stack_traces.py task/ output.csv --only-with-solutions")
        print("\nExtracts stack traces AND solutions from developer log files.")
        print("Output CSV columns: id, stack_trace, solution")
        print("\nOptions:")
        print("  --only-with-solutions  Only save traces that have solutions (filters out traces without solutions)")
        sys.exit(1)

    task_dir = sys.argv[1]

    # Check for --only-with-solutions flag
    only_with_solutions = '--only-with-solutions' in sys.argv

    # Get output_csv (handle both positions)
    if len(sys.argv) > 2 and not sys.argv[2].startswith('--'):
        output_csv = sys.argv[2]
    else:
        output_csv = "stack_traces_with_solutions.csv"

    if not Path(task_dir).exists():
        print(f"Error: Directory not found: {task_dir}")
        sys.exit(1)

    extract_all_stack_traces(task_dir, output_csv, only_with_solutions=only_with_solutions)


if __name__ == "__main__":
    main()
