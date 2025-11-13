"""
Script to convert response.json forecast data into a well-formatted Excel file
"""

import json
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
from datetime import datetime

def load_json_data(file_path):
    """Load JSON data from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_dataframe(data):
    """Create a combined DataFrame from the JSON structure"""
    rows = []
    
    forecasts = data.get('forecasts', {})
    from_last = forecasts.get('from_last_data_point', {})
    from_fy = forecasts.get('from_fy_start', {})
    
    # Get all branches
    all_branches = set(from_last.keys()) | set(from_fy.keys())
    
    # Get all dates from both sources
    all_dates = set()
    for branch_data in from_last.values():
        for item in branch_data:
            all_dates.add(item['date'])
    for branch_data in from_fy.values():
        for item in branch_data:
            all_dates.add(item['date'])
    
    all_dates = sorted(list(all_dates))
    
    # Create a dictionary for quick lookup
    from_last_dict = {}
    for branch, branch_data in from_last.items():
        for item in branch_data:
            key = (branch, item['date'])
            from_last_dict[key] = item
    
    from_fy_dict = {}
    for branch, branch_data in from_fy.items():
        for item in branch_data:
            key = (branch, item['date'])
            from_fy_dict[key] = item
    
    # Create rows
    for branch in sorted(all_branches):
        for date in all_dates:
            row = {
                'Branch': branch,
                'Date': date
            }
            
            # Add from_last_data_point data
            key = (branch, date)
            if key in from_last_dict:
                item = from_last_dict[key]
                row['Quantity (from_last_data_point)'] = item.get('quantity', '')
                row['Type (from_last_data_point)'] = item.get('type', '')
            else:
                row['Quantity (from_last_data_point)'] = ''
                row['Type (from_last_data_point)'] = ''
            
            # Add from_fy_start data
            if key in from_fy_dict:
                item = from_fy_dict[key]
                row['Quantity (from_fy_start)'] = item.get('quantity', '')
                row['Type (from_fy_start)'] = item.get('type', '')
            else:
                row['Quantity (from_fy_start)'] = ''
                row['Type (from_fy_start)'] = ''
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

def format_excel(file_path):
    """Apply formatting to the Excel file"""
    wb = load_workbook(file_path)
    ws = wb.active
    
    # Define styles
    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF", size=11)
    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    center_align = Alignment(horizontal='center', vertical='center')
    
    # Format header row
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = center_align
        cell.border = border
    
    # Format data rows
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for cell in row:
            cell.border = border
            if cell.column in [1, 2]:  # Branch and Date columns
                cell.alignment = Alignment(vertical='center')
            elif cell.column > 2:  # Numeric columns
                cell.alignment = Alignment(horizontal='right', vertical='center')
                if cell.value and isinstance(cell.value, (int, float)):
                    cell.number_format = '#,##0.0'
    
    # Auto-adjust column widths
    for column in ws.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 50)
        ws.column_dimensions[column_letter].width = adjusted_width
    
    # Freeze header row
    ws.freeze_panes = 'A2'
    
    wb.save(file_path)

def main():
    """Main function to create Excel file"""
    input_file = 'response.json'
    output_file = 'forecast_data.xlsx'
    
    print(f"Loading data from {input_file}...")
    data = load_json_data(input_file)
    
    print("Creating DataFrame...")
    df = create_dataframe(data)
    
    print(f"Writing to {output_file}...")
    df.to_excel(output_file, index=False, sheet_name='Forecasts', engine='openpyxl')
    
    print("Applying formatting...")
    format_excel(output_file)
    
    print(f"✓ Excel file created successfully: {output_file}")
    print(f"  Total rows: {len(df)}")
    print(f"  Branches: {df['Branch'].unique().tolist()}")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")

if __name__ == '__main__':
    main()

