"""
UI Styles and Theming

Centralized styling for the AxonVision Camera Calibration Tool.
"""

# Color palette
COLORS = {
    'primary': '#2E86AB',        # Blue - headers, icons
    'primary_dark': '#1B5E7D',   # Darker blue
    'success': '#28A745',        # Green - passed, success buttons
    'success_hover': '#218838',  # Darker green for hover
    'warning': '#FFC107',        # Yellow/Orange - warnings
    'danger': '#DC3545',         # Red - remove buttons, errors
    'danger_hover': '#C82333',   # Darker red for hover
    'text_dark': '#2C3E50',      # Dark text
    'text_muted': '#6C757D',     # Muted/secondary text
    'background': '#F8F9FA',     # Light gray background
    'white': '#FFFFFF',
    'border': '#DEE2E6',         # Border color
    'table_header': '#E3F2FD',   # Light blue for table headers
    'table_row_alt': '#F8F9FA',  # Alternating row color
}

# Main application stylesheet
MAIN_STYLESHEET = f"""
QMainWindow {{
    background-color: {COLORS['background']};
}}

QWidget {{
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 14px;
    color: {COLORS['text_dark']};
}}

QLabel {{
    color: {COLORS['text_dark']};
}}

QLabel#title {{
    font-size: 32px;
    font-weight: bold;
    color: {COLORS['primary']};
}}

QLabel#subtitle {{
    font-size: 16px;
    color: {COLORS['text_muted']};
}}

QLabel#section_header {{
    font-size: 18px;
    font-weight: bold;
    color: {COLORS['text_dark']};
    padding-top: 10px;
}}

QLabel#screen_indicator {{
    font-size: 12px;
    color: {COLORS['text_muted']};
    font-style: italic;
}}

QLabel#step_indicator {{
    font-size: 12px;
    color: {COLORS['text_muted']};
}}

QLabel#version {{
    font-size: 11px;
    color: {COLORS['text_muted']};
}}

QPushButton {{
    padding: 10px 20px;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 500;
    border: none;
}}

QPushButton#primary_button {{
    background-color: {COLORS['success']};
    color: white;
    padding: 15px 40px;
    font-size: 18px;
    font-weight: bold;
    min-width: 300px;
}}

QPushButton#primary_button:hover {{
    background-color: {COLORS['success_hover']};
}}

QPushButton#secondary_button {{
    background-color: {COLORS['white']};
    color: {COLORS['primary']};
    border: 2px solid {COLORS['primary']};
    padding: 15px 40px;
    font-size: 16px;
    font-weight: 500;
    min-width: 300px;
}}

QPushButton#secondary_button:hover {{
    background-color: {COLORS['table_header']};
}}

QPushButton#settings_button {{
    background-color: {COLORS['background']};
    color: {COLORS['text_muted']};
    border: 1px solid {COLORS['border']};
    padding: 8px 16px;
    font-size: 13px;
}}

QPushButton#settings_button:hover {{
    background-color: {COLORS['white']};
    border-color: {COLORS['text_muted']};
}}

QPushButton#add_button {{
    background-color: {COLORS['primary']};
    color: white;
    padding: 8px 16px;
    font-size: 13px;
}}

QPushButton#add_button:hover {{
    background-color: {COLORS['primary_dark']};
}}

QPushButton#remove_button {{
    background-color: {COLORS['danger']};
    color: white;
    padding: 6px 12px;
    font-size: 12px;
}}

QPushButton#remove_button:hover {{
    background-color: {COLORS['danger_hover']};
}}

QPushButton#verify_button {{
    background-color: {COLORS['warning']};
    color: {COLORS['text_dark']};
    padding: 6px 12px;
    font-size: 12px;
    font-weight: bold;
}}

QPushButton#verify_button:hover {{
    background-color: #E0A800;
}}

QPushButton#nav_button {{
    background-color: {COLORS['primary']};
    color: white;
    padding: 10px 24px;
    font-size: 14px;
}}

QPushButton#nav_button:hover {{
    background-color: {COLORS['primary_dark']};
}}

QPushButton#cancel_button {{
    background-color: {COLORS['text_muted']};
    color: white;
    padding: 10px 24px;
    font-size: 14px;
}}

QPushButton#cancel_button:hover {{
    background-color: #5A6268;
}}

QLineEdit {{
    padding: 8px 12px;
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    background-color: {COLORS['white']};
}}

QLineEdit:focus {{
    border-color: {COLORS['primary']};
}}

QComboBox {{
    padding: 8px 12px;
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    background-color: {COLORS['white']};
    min-width: 100px;
}}

QComboBox:focus {{
    border-color: {COLORS['primary']};
}}

QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {COLORS['text_muted']};
    margin-right: 8px;
}}

QTableWidget {{
    background-color: {COLORS['white']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    gridline-color: {COLORS['border']};
}}

QTableWidget::item {{
    padding: 8px;
}}

QHeaderView::section {{
    background-color: {COLORS['table_header']};
    color: {COLORS['primary']};
    font-weight: bold;
    padding: 10px;
    border: none;
    border-bottom: 2px solid {COLORS['border']};
}}

QScrollArea {{
    border: none;
    background-color: transparent;
}}

QFrame#card {{
    background-color: {COLORS['white']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 20px;
}}

QFrame#recent_item {{
    background-color: {COLORS['white']};
    border-bottom: 1px solid {COLORS['border']};
    padding: 12px 16px;
}}

QFrame#recent_item:hover {{
    background-color: {COLORS['table_row_alt']};
}}
"""

# Status label styles
def get_status_style(status: str) -> str:
    """Get style for calibration status label."""
    if status.lower() == 'passed':
        return f"color: {COLORS['success']}; font-weight: bold;"
    elif status.lower() == 'warning':
        return f"color: {COLORS['warning']}; font-weight: bold;"
    elif status.lower() == 'failed':
        return f"color: {COLORS['danger']}; font-weight: bold;"
    return f"color: {COLORS['text_muted']};"
